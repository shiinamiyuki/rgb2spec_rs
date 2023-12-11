use rgb2spec::*;
use std::io::Write;
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        println!(
            "Syntax: rgb2spec_opt <resolution> <output> <gamut>\n\
        where <gamut> is one of \
        sRGB,eRGB,XYZ,ProPhotoRGB,ACES2065_1,REC2020\n"
        );
        std::process::exit(1);
    }
    let gamut = match args[3].as_str() {
        "sRGB" => Gamut::SRgb,
        "eRGB" => Gamut::ERgb,
        "XYZ" => Gamut::Xyz,
        "ProPhotoRGB" => Gamut::ProPhotoRgb,
        "ACES2065_1" => Gamut::Aces2065_1,
        "REC2020" => Gamut::Rec2020,
        "DCI_P3" => Gamut::DciP3,
        _ => {
            println!("Unknown gamut {}", args[3]);
            std::process::exit(1);
        }
    };
    let cs = RgbColorSpace::from_gamut(gamut);
    let table = optimize(&cs, 64);
    let mut file = std::fs::File::create("out.spec").unwrap();
    let data = unsafe {
        std::slice::from_raw_parts(
            table.data.as_ptr() as *const u8,
            table.data.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(data).unwrap();
}
