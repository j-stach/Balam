use crate::utils::*;

pub trait Output {
    fn classify(&self) -> Vec<usize>;
    //fn cce_loss() -> 
}
impl Output for Vec<Vec<f32>> {
    /// Determines a result from a probability distribution
    fn classify(&self) -> Vec<usize> {
        let classification = self.iter().map(|s| s.argmax()).collect::<Vec<usize>>();
        return classification
    }
}