/// Trims away negative values for ReLU activation
pub fn take_pos(val: f32) -> f32 {
    let mut val = val;
    if val < 0. { val = 0.; }
    return val
}

/// Math functions for f32 vecs
pub trait MathUtils {
    fn dot_product(&self, rhs: &Vec<f32>) -> f32;
    fn sum(&self) -> f32;
    fn argmax(&self) -> usize;
}
impl MathUtils for Vec<f32> {
    /// Dot product of two vectors of f32
    fn dot_product(&self, rhs: &Vec<f32>) -> f32 {
        assert_eq![self.len(), rhs.len()];
        let mut product: f32 = 0.;
        for i in 0..self.len() {
            product += self[i] * rhs[i];
        }
        return product
    }
    /// Add together all elements of a vector of f32
    fn sum(&self) -> f32 {
        let mut sum: f32 = 0.;
        for f in self { sum += f };
        return sum
    }
    /// Find the index of the greatest f32 in the vector
    // If you're looking for a max when your vec is empty, your program deserves to crash.
    fn argmax(&self) -> usize {
        let argmax = self.iter()
                         .enumerate()
                         .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                         .map(|(i, _)| i).unwrap();
        return argmax
    }
}
