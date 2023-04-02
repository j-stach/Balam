// When optimizing performance, implement ndarray's dot or add rayon par_iter to the dot trait below
// And change matrix to use arrays instead of vectors
use crate::layer::*;
use crate::output::*;

mod layer;
mod activation;
mod utils;
mod output;

fn main() {

    two_dense_layer();

}

fn two_dense_layer() {

    let sample_batch: Vec<Vec<f32>> = vec![
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
    ]; 
    let targets: Vec<usize> = vec![0, 0, 0];
    
    let layer0 = DenseLayer::new(4, 3);
    println!("Hidden Layer:");
    for node in layer0.0.iter() {
        println!("bias: {}", node.bias);
        print!("weights: ");
        for weight in &node.weights {
            print!("{weight} ");
        }
        print!("\n");
    }
    let layer1 = DenseLayer::new(3, 3);
    println!("Output Layer:");
    for node in layer1.0.iter() {
        println!("bias: {}", node.bias);
        print!("weights: ");
        for weight in &node.weights {
            print!("{weight} ");
        }
        print!("\n");
    }

    let layer0_outputs: Vec<Vec<f32>> = layer0.forward(&sample_batch);
    let layer1_outputs: Vec<Vec<f32>> = layer1.predict(&layer0_outputs);
    println!("Outputs:");
    for outputs in &layer1_outputs { 
        for output in outputs { print!("{output} ") };
        print!("\n");
    };

    let classes = &layer1_outputs.classify();
    println!("Classes:");
    for result in classes { println!("{result} ") };

    let losses = cat_loss(&layer1_outputs, targets);
    println!("Losses:");
    for loss in losses { println!("{loss} ") };
}

/// Categorical cross-entropy loss calculation
// After classification of output, compare the target to the output
// Then run log loss on the f32 value at the target index to determine inaccuracy
fn cat_loss(outputs: &Vec<Vec<f32>>, targets: Vec<usize>) -> Vec<f32> {
    // For each in target class, get the usize and the index, 
    // Then use the index to id the output set, then the usize to get the correct index of outputs
    // Finally, get the natural log of the f32 value at that location
    let mut losses: Vec<f32> = Vec::with_capacity(targets.len());
    for (c, class) in targets.iter().enumerate() {
        losses.push(outputs[c][*class].ln());
    }
    return losses
}






