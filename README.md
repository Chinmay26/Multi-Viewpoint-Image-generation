# Multi-Viewpoint-Image-generation
This project investigates three deep convolutional architectures to generate multiple views from a given single view of an object in an arbitrary pose. Traditional CV techniques can handle affine transformations such as rotation, translation, scaling well. However, generating non-affine transformations like view-point change, projective transformations from single 2D images is a challenging task. The challenges may be due to unspecified viewing angle,  partial object occlusion, 3D object shape ambiguity, pose ambiguity.

Image generation and transformations tasks have many practical applications in robotics and computer visions. Rendering multiple 2D views is helpful in generating 3D representation of that object.  In robotics, generating multiple views can help in better grasping of objects by giving them a better understanding of hidden parts of object. It can also be used as a pre-processing step in vision algorithms such as Image classification/labeling, face verification to check for duplicate viewpoint images.


### Problem Statement
Generate synthetic 2D images of an object in a target viewpoint from a single input image.


### Introduction:
The problem is approached in three phases
- Phase 1 tackles generating image in target viewpoint which was implicit to the model [target viewpoint is always 30 deg counterclockwise].
- Phase 2 tackles generating synthetic 2D images given any target viewpoint [azimuth angle].
- Phase 3 investigates different adversarial training models (Vanilla GAN, DCGAN) to generate realistic synthetic views, given any viewpoint. 


## Dataset
- [ShapeNet dataset](https://shapenet.org/). The focus is on three model categories: Car, Chair and Mug.
- The [2D images](https://drive.google.com/drive/folders/0Byb88ed56z69LWVJWWRIRVQ0Rkk?usp=sharing) of the above 3D models are rendered using [ShapeNet Renderer](https://github.com/ShapeNet/shapenet-viewer). Both input and output images are of resolution 64 by 64 by 3.
For each model, images are rendered from 36 viewpoints corresponding to 36 azimuth angles and 0 elevation angle. 
- 80:20 training/test split is used. For Car category, the total 3512 models is plit into 2809 models [98315 images ] for training and the remaining 703 models [24605 images] for testing. 
- For Chair category,  from total 6775 models, we use 5402 models [189700 images] and the remaining 1355 models [ 47425 images ] for testing. We used the Mug category for prototyping.

## Loss
We optimise on the L1 loss function to improve the synthetic image detail and quality . We experimented with different loss functions such as Local Moment Loss, L2 loss, perceptual loss (pre-trained deep networks), but got best results with L1 loss.

## Normalization and Non-linearity
All input images are nomralized to [-1,1] and LeakyRelu is used for non-linearity in intermediate layers and Tanh for the final layer. We experimented with other combinations (Sigmoid / ReLU) but got best results for above combination

## Models
1. **Autoencoder**
   -  Architecture
   
   Autoencoders have been known to perform well in capturing abstract information of image contents and can recover the original image by upsampling the feature maps. Therefore, we designed a baseline Autoencoder to generate 2D image in a different pose as that of given input image
   
   ![AE Arch](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/vanilla_ae.png?raw=true)
   
                                                   Encoder Architecture
![Encoder Results: Fig 1](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/car_ae_wo_pose.png?raw=true)

                                                   Encoder Results: Fig 1 
                                                   
![AE Loss](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/car_ae_without_pose_.PNG?raw=true)

                                                   Encoder L1 Loss Curves

   -  Encoder Results
      -  Results Interpretation: Fig 1 shows the results of the Vanilla AE.  From Fig 1, for car models 1 and 4, the input and target views are close. Thus, AE works well since there is not large viewpoint transformation change. Model 6 gives us a deformed result since the input image has its front view hidden.
      -  The above baseline model when trained on mug dataset, was unable to produce an accurate output in target viewpoint. For example, when testing it on mug dataset, it couldn’t reproduce mug handle in target viewpoint. This is due to that baseline model failed to learn the pose information.
   -  Experimental findings
      -  Pooling: Max pooling reduces the number of dimensions of input to reduce computation costs, but in our case, the model couldn’t learn abstract features of image because of max pooling. We decided to remove max pooling from above model and replaced tanh functions by ReLU function except the last deconvolutional layer.
      -  Activation Function: ReLU function is useful as it prevents gradients from saturating. However, ReLU suffers from a problem wherein ReLU units die during training and if that happens, gradient through that point will be zero forever. To resolve this issue, ReLU was replaced by Leaky ReLU
   -  Improvement area:
      -  The vanilla AE cannot generate synthetic views in any given target viewpoint. Encoding pose information to model explicitly will help in generating synthetic views in target viewpoint


2. **Pose Encoder**
   -  The vanilla AE model has no explicit understanding of the target viewpoint. The next step is incorporate pose information into our model. We represented pose as a 36D one-hot vector corresponding to azimuth angles from [0-350]. The pose is broadcasted as a cube and then concatenated with the latent space. The pose signal considerably improves the quality of the synthetic views.

   -  Architecture
![AE Arch](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/pose_encoder.png?raw=true)

                                             PoseEncoder Architecture

   -  Pose Encoder Results
![Encoder Results](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/car_ae_with_pose.png?raw=true)
                                            
                                             PoseEncoder Results: Fig 2
                                            

![AE Loss](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/car_ae_with_pose_.PNG?raw=true)

                                             PoseEncoder L1 Loss

   -  Result Interpretation: The pose encoder performs considerably better than the vanilla AE. The results from Fig 2 show some results from random input-output pair combinations. It preserves the structural property of the input object. It performs best when there is small transformation change [< 180 deg] . This is expected since the Pose Encoder has to **perdict** the pixels for the occluded regions. For large transformations, there is larger occluded area between the input view and target view. This explains the larger L1 test loss [~0.09] for test models with larger view-point transformation.

   -  Experimental Findings
      - Architecture: We experimented with different deep architectures and 3-4 intermediate layers, different filter dimensions. Our experiments revealed that L1 loss does not show any improvement when architecture was beyond 3  intermediate layers. Adding final FC layers degraded the performance of the model.

      -  Pose Signal: This was a **challenging part** since we can embed pose in different ways: one-hot vector, cube, amplified vector etc. We experimented with the above and got best general results across categories for the above representation. There is also a need to balance the pose signal with the image signal. The pose signal should not overshadow the image content in the latent space. As such, we used a 4 * 4 * 36 cube for pose signal representation and 4*4*92 for the image content.
Hyperparameters: Adding Batch Norm between intermediate layers helped to converge faster. Tweaking learning rates did not help to a larger extent.
      -  Improvement Areas
The state-of-the art models use a flow field or a visibility map to aid image generation. This helps to separate out the regions where pixels need to be relocated with pixels which need to be predicted. Adding such a representation to the model may help to further improve the quality of the images.

  3. **Pose Encoder with Adversarial Training**
GAN models have been used to generate realistic high quality images. The Pose Encoder preserves image structure but is unable to generate rich textures and high quality images. We combined them to have a joint loss function and train the model in an adversarial fashion. We use DCGAN architecture for reference.
   -  Architecture
         ![Generator](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/generator.png?raw=true)
         
                                                  Generator  
         ![Discriminator](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/discriminator.png?raw=true)
         
                                                  Discriminator
   
         * Combined Loss:  Alpha * L1 loss + Beta * GAN loss *
         We used adversarial training and jointly trained the model. Alpha = 1.0 and Beta = 0.2 gave best results. Here, we were more focused on the quality of the output images rather than the combined  loss values.

                                                      
        
![Gan Results](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/car_gan.png?raw=true)

                                                            Fig 3

![GAN Discriminator Loss](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/car_gan_dis_loss.PNG?raw=true)

                                                      Combined Generator Loss

![GAN Generator Loss](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation/blob/master/images/car_gan_gen_loss.PNG?raw=true)

                                                      Discriminator Loss

   -  Results:
From Fig 3, we can see that we are able to generate better quality images in comparison to . In model 3, even when there is a large viewpoint change [180 deg], we are able to generate good quality images. With more training, we can generate highly detailed images.

   -  Experiments:  **Training difficulties and tricks used to balance the min-max game**
   GANs are volatile and **highly unstable** during training. In our case, the discriminator was very strong and the discriminator loss [especially **dreal_loss**] dropped to 0 after few epochs. We had to balance out the power and we used the following tricks:
   -  Feature Matching: Instead of Generator minimizing the output of D, we trained G to maximize the L2 loss of an intermediate activation layer of D.
   -  Make Generator stronger: Generator is update more frequently [5 vs 1] than the discriminator. Higher learning rate [5e-4 vs 5e-5] applied to Generator than Discriminator.
Make Discriminator weaker: Remove batch norm from discriminator. Add higher dropout rate [0.5] to slow down convergence of discriminator.
   -  Monitor via Loss statistics: Stop training the discriminator if its loss falls below the a threshold. [dreal_loss < 0.2 , dfake_loss < 0.2]
      -  Using Soft and Noisy Labels: Add some perturbation to label values. Replace Real = 1 to Real = [0.8, 1.2]. We only did it for Real labels since only dreal_loss was dropping to 0 in few epochs.
   -  Improvements
      -  The discriminator is currently rudimentary. It only has to distinguish between fake and real samples. We can extend the difficulty of the discriminator by enforcing the discriminator to distinguish between correct vs incorrect poses + correct vs incorrect image labels. 
      -  Train for longer hours. GANs have proven to improve image quality when trained for several days. Due to time constraint, we only ran it for ~2 hours.

## Comparison to state-of-art techniques

There has been previous works done in view-synthesis notably [Flow-field networks](https://arxiv.org/abs/1605.03557) and [Transformation Grounded Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Park_Transformation-Grounded_Image_Generation_CVPR_2017_paper.pdf). We compare our model with these state-of the art models.

|          Single-view                   |    L1 loss    |        Input Image      |
| -------------------------------------- |:-------------:| ------------------------|
| Flow Field Networks                    |  0.368        |    224 * 224 * 3        |
| Transformation Grounded Networks       |  0.132        |    256 * 256 * 3        |
| Ours                                   | 0.0514        |     64  * 64  * 3       |

## Future Work: 
1. Extend the difficulty of the discriminator: Enforce it to distinguish correct vs incorrect poses + correct vs incorrect models.
2. Experiment with real world objects and scenes. Other datasets such as KiTTI.
3. Explore Capsule Nets. Can Capsule Nets be used with adversarial training for image generation?

## Major Challenges encountered: 
1. Pose Encoding: How can we encode the pose information to the model? Which representation helps in image generation?
2. GAN training: GANs are highly volatile. It took considerable research and hand-tuning to ensure the min-max game between the generator and discriminator. Thanks to several online resources, we were able to prevent the discriminator from dropping to failure mode [loss = 0]. 
3. Memory handling: We were dealing with image datasets of worth >3GB. Due to a bug in our code which resulted in float64 conversion, we ended up 3GB * 8 * 3 ~ 70 GB. This bloated memory caused us a minor setback initially during training.


## References:
1. http://openaccess.thecvf.com/content_cvpr_2017/papers/Park_Transformation-Grounded_Image_Generation_CVPR_2017_paper.pdf
2. https://arxiv.org/abs/1605.03557
3. https://github.com/soumith/ganhacks
4. https://github.com/skaae/vaeblog
5. https://theneuralperspective.com/2016/11/12/improved-techniques-for-training-gans/
6. https://github.com/aleju/papers/blob/master/neural-nets/Improved_Techniques_for_Training_GANs.md
7. https://github.com/shaohua0116/DCGAN-Tensorflow
8. https://github.com/gitlimlab/Generative-Latent-Optimization-Tensorflow


