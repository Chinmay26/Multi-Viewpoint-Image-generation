# Multi-Viewpoint-Image-generation
We investigate three deep convolutional architectures to generate multiple views from a given single view of an object in an arbitrary pose. Traditional CV techniques can handle affine transformations such as rotation, translation, scaling well. However, generating non-affine transformations like view-point change, projective transformations from single 2D images is a challenging task. The challenges may be due to unspecified viewing angle,  partial object occlusion, 3D object shape ambiguity, pose ambiguity.
Image generation and transformations tasks have many practical applications in robotics and computer visions. Rendering multiple 2D views is helpful in generating 3D representation of that object.  In robotics, generating multiple views can help in better grasping of objects by giving them a better understanding of hidden parts of object. It can also be used as a pre-processing step in vision algorithms such as Image classification/labeling, face verification to check for duplicate viewpoint images.


### Problem Statement
Generate synthetic 2D images of an object in a target viewpoint from a single input image.
