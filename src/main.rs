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

    let predictions = &layer1_outputs.classify();
    println!("Classes:");
    for result in predictions { println!("{result} ") };

    let losses = cat_loss(&layer1_outputs, &targets);
    println!("Losses:");
    for loss in losses { println!("{loss} ") };

    let accuracy = accuracy(predictions, &targets);
    println!("Accuracy = {accuracy}");
}

/// Categorical cross-entropy loss calculation
fn cat_loss(outputs: &Vec<Vec<f32>>, targets: &Vec<usize>) -> Vec<f32> {
    let mut losses: Vec<f32> = Vec::with_capacity(targets.len());
    for (c, class) in targets.iter().enumerate() {
        let mut output = outputs[c][*class];
        if output == 0. { output = 0.000000001 };
        losses.push(-output.ln());
    }
    return losses
}

/// Calculate batch accuracy
fn accuracy(predictions: &Vec<usize>, targets: &Vec<usize>) -> f32 {
    assert_eq![predictions.len(), targets.len()];
    let mut misses: usize = 0;
    let comparison = predictions.iter().zip(targets.iter());
    for (p, t) in comparison {
        if p != t { misses += 1 }
    }
    let accuracy: f32 = (predictions.len() as f32 - misses as f32) / targets.len() as f32;
    return accuracy
}


