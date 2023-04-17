use crate::utils::*;

/// Activation patterns
pub trait Activation {
    fn relu(&self) -> Vec<Vec<f32>>;
    fn softmax(&self) -> Vec<Vec<f32>>;
}
impl Activation for Vec<Vec<f32>> {
    /// Rectified linear activation pattern, for hidden layers
    fn relu(&self) -> Vec<Vec<f32>> {
        self.iter()
            .map(|os| { 
                os.iter()
                  .map(|o| {
                        take_pos(*o)
                  }).collect::<Vec<f32>>()
            }).collect::<Vec<Vec<f32>>>()
    }
    /// Softmax exponential activation function, for the output layer in classification models
    // Exploding value risk still needs to be handled? (Only if input values are large?)
    fn softmax(&self) -> Vec<Vec<f32>> {
        let e: f32 = 2.718282;
        // Sum each sample after applying exponents
        let output_sum: Vec<f32> = self.iter()
                                  .map(|os| { 
                                      os.iter()
                                        .map(|o| {
                                            e ** o
                                        }).collect::<Vec<f32>>().sum()
                                  }).collect::<Vec<f32>>();
        // Run it again to generate probability distribution
        let outputs: Vec<Vec<f32>> = self.iter()
                                         .enumerate()
                                         .map(|(i, os)| { 
                                             os.iter()
                                               .map(|o| {
                                                   (e ** o) / output_sum[i]
                                               }).collect::<Vec<f32>>()
                                         }).collect::<Vec<Vec<f32>>>();
        return outputs
    }
}
