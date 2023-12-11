use super::*;
/// LU decomposition with partial pivoting.
pub(crate) trait Lu {
    type Permutation;
    type Vector;
    fn precompute(&mut self, tol: f64) -> Option<Self::Permutation>;

    fn solve(&self, p: Self::Permutation, b: Self::Vector) -> Self::Vector;
}
impl Lu for [[f64; 3]; 3] {
    type Vector = [f64; 3];
    type Permutation = [usize; 4];
    fn precompute(&mut self, tol: f64) -> Option<[usize; 4]> {
        const N: usize = 3;
        let mut p: [usize; 4] = array::from_fn(|i| i);

        for i in 0..N {
            let mut max_a = 0.0;
            let mut i_max = i;

            for k in i..N {
                let abs_a = self[k][i].abs();
                if abs_a > max_a {
                    max_a = abs_a;
                    i_max = k;
                }
            }
            if max_a < tol {
                return None;
            }

            if i_max != i {
                p.swap(i, i_max);
                self.swap(i, i_max);

                p[N] += 1;
            }

            for j in (i + 1)..N {
                self[j][i] /= self[i][i];
                for k in (i + 1)..N {
                    self[j][k] -= self[j][i] * self[i][k];
                }
            }
        }
        Some(p)
    }
    fn solve(&self, p: Self::Permutation, b: Self::Vector) -> Self::Vector {
        const N: usize = 3;
        let mut x: [f64; 3] = [0.0; 3];
        for i in 0..N {
            x[i] = b[p[i]];
            for k in 0..i {
                x[i] -= self[i][k] * x[k];
            }
        }
        for i in (0..N).rev() {
            for k in i + 1..N {
                x[i] -= self[i][k] * x[k];
            }
            x[i] /= self[i][i];
        }
        x
    }
}

fn eval_jacobian(cs: &RgbColorSpace, coeffs: [f64; 3], rgb: [f64; 3]) -> [[f64; 3]; 3] {
    let mut jac = [[0.0; 3]; 3];
    for i in 0..3 {
        let mut tmp = coeffs;
        tmp[i] -= RGB2SPEC_EPSILON;
        let r0 = eval_residual(cs, tmp, rgb);

        let mut tmp = coeffs;
        tmp[i] += RGB2SPEC_EPSILON;
        let r1 = eval_residual(cs, tmp, rgb);

        for j in 0..3 {
            jac[j][i] = (r1[j] - r0[j]) / (2.0 * RGB2SPEC_EPSILON);
        }
    }
    jac
}

fn eval_residual(cs: &RgbColorSpace, coeffs: [f64; 3], rgb: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..CIE_FINE_SAMPLES {
        let lambda = (cs.lambda_tbl[i] - CIE_LAMBDA_MIN) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
        let mut x = 0.0;
        for i in 0..3 {
            x = x * lambda + coeffs[i];
        }
        let s = sigmoid(x);
        for j in 0..3 {
            out[j] += cs.rgb_tbl[j][i] * s;
        }
    }
    cie_lab(cs, &mut out);
    let mut residual = rgb;
    cie_lab(cs, &mut residual);
    for j in 0..3 {
        residual[j] -= out[j];
    }
    residual
}

#[allow(non_snake_case)]
fn cie_lab(cs: &RgbColorSpace, p: &mut [f64; 3]) {
    let mut X = 0.0;
    let mut Y = 0.0;
    let mut Z = 0.0;
    let Xw = cs.xyz_whitepoint[0];
    let Yw = cs.xyz_whitepoint[1];
    let Zw = cs.xyz_whitepoint[2];
    for j in 0..3 {
        X += p[j] * cs.to_xyz[0][j];
        Y += p[j] * cs.to_xyz[1][j];
        Z += p[j] * cs.to_xyz[2][j];
    }

    let f = |t: f64| -> f64 {
        let delta = 6.0 / 29.0;
        if t > delta * delta * delta {
            t.cbrt()
        } else {
            t / (delta * delta * 3.0) + (4.0 / 29.0)
        }
    };
    p[0] = 116.0 * f(Y / Yw) - 16.0;
    p[1] = 500.0 * (f(X / Xw) - f(Y / Yw));
    p[2] = 200.0 * (f(Y / Yw) - f(Z / Zw));
}

fn gauss_newton(cs: &RgbColorSpace, rgb: [f64; 3], coeffs: &mut [f64; 3], it: usize) -> bool {
    let mut r;
    let mut residuals = vec![];
    let mut jacobians = vec![];
    for _ in 0..it {
        let residual = eval_residual(cs, *coeffs, rgb);
        let jac = eval_jacobian(cs, *coeffs, rgb);
        residuals.push(residual);
        jacobians.push(jac);

        let mut jac = eval_jacobian(cs, *coeffs, rgb);
        let p = jac.precompute(1e-15);
        if p.is_none() {
            return false;
        }
        let p = p.unwrap();
        let x = jac.solve(p, residual);

        r = 0.0;

        for j in 0..3 {
            coeffs[j] -= x[j];
            r += residual[j] * residual[j];
        }
        let max = coeffs[0].max(coeffs[1]).max(coeffs[2]);
        if max > 200.0 {
            for j in 0..3 {
                coeffs[j] *= 200.0 / max;
            }
        }
        if r < 1e-6 {
            break;
        }
    }
    true
}
#[derive(Clone, Copy)]
#[repr(transparent)]
struct SendPtr<T>(*mut T);
impl<T> SendPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
fn parallel_for<F: Fn(usize) + Send + Sync>(count: usize, f: F) {
    let cnt = AtomicUsize::new(0);
    let num_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let chunk_size = (count / (8 * num_threads)).max(1);
    let threads = (0..num_threads)
        .map(|_| {
            let cnt: &'static AtomicUsize = unsafe { std::mem::transmute(&cnt) };
            let f = &f as *const F as usize;
            std::thread::spawn(move || loop {
                let f = unsafe { std::mem::transmute::<usize, &F>(f) };
                let i = cnt.fetch_add(chunk_size, Ordering::Relaxed);
                if i >= count {
                    break;
                }
                for j in i..(i + chunk_size).min(count) {
                    f(j);
                }
            })
        })
        .collect::<Vec<_>>();
    threads.into_iter().for_each(|t| t.join().unwrap());
}
pub fn optimize(cs: &RgbColorSpace, res: usize) -> Rgb2SpecTable {
    let scale = (0..res)
        .map(|k| smoothstep(smoothstep(k as f64 / (res - 1) as f64)) as f32)
        .collect::<Vec<f32>>();

    let buf_size = 3 * 3 * res * res * res;
    let mut out = vec![0.0; buf_size];
    let it = 15;

    for l in 0..3 {
        let out_ptr = SendPtr(out.as_mut_ptr());
        parallel_for(res, |j| {
            let out = unsafe { std::slice::from_raw_parts_mut(out_ptr.get(), buf_size) };
            let y = j as f64 / (res - 1) as f64;
            for i in 0..res {
                let x = i as f64 / (res - 1) as f64;
                let mut coeffs = [0.0; 3];
                let mut rgb = [0.0; 3];

                let start = res / 5;

                for k in start..res {
                    let b = scale[k] as f64;
                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    if !gauss_newton(cs, rgb, &mut coeffs, it) {
                        println!(
                            "solve failed for l:{} i:{}, j:{}, k:{}, rgb:{:.5?}",
                            l, i, j, k, rgb
                        );
                    }

                    let c0 = CIE_LAMBDA_MIN;
                    let c1 = 1.0 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);

                    let a = coeffs[0];
                    let b = coeffs[1];
                    let c = coeffs[2];

                    let idx = ((l * res + k) * res + j) * res + i;

                    out[3 * idx + 0] = (a * sqr(c1)) as f32;
                    out[3 * idx + 1] = (b * c1 - 2.0 * a * c0 * sqr(c1)) as f32;
                    out[3 * idx + 2] = (c - b * c0 * c1 + a * sqr(c0 * c1)) as f32;
                }

                coeffs.fill(0.0);

                for k in (0..=start).rev() {
                    let b = scale[k] as f64;

                    rgb[l] = b;
                    rgb[(l + 1) % 3] = x * b;
                    rgb[(l + 2) % 3] = y * b;

                    if !gauss_newton(cs, rgb, &mut coeffs, it) {
                        println!(
                            "solve failed for l:{} i:{}, j:{}, k:{}, rgb:{:.5?}",
                            l, i, j, k, rgb
                        );
                    }

                    let c0 = CIE_LAMBDA_MIN;
                    let c1 = 1.0 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);

                    let a = coeffs[0];
                    let b = coeffs[1];
                    let c = coeffs[2];

                    let idx = ((l * res + k) * res + j) * res + i;

                    out[3 * idx + 0] = (a * sqr(c1)) as f32;
                    out[3 * idx + 1] = (b * c1 - 2.0 * a * c0 * sqr(c1)) as f32;
                    out[3 * idx + 2] = (c - b * c0 * c1 + a * sqr(c0 * c1)) as f32;
                }
            }
        });
    }
    Rgb2SpecTable {
        data: out,
        res,
        scale,
    }
}
