use std::{fmt::Display, fs, path::Path};

pub fn fs_write_displayable<P: AsRef<Path>, T: Display>(path: P, v: &T) {
    fs::write(path, format!("{v}")).expect("filesystem operation failed");
}

pub struct Spectra<'a>(pub f64, pub &'a [f64]);
impl<'a> Display for Spectra<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\t{}", self.0)?;
        for v in self.1 {
            writeln!(f, "{v}")?;
        }
        Ok(())
    }
}
