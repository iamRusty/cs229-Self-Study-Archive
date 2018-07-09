# cs229 Self-study Guide
Lecture videos used are from the cs229 2008 class. All other course materials are up-to-date. 
The original cs229 course uses Matlab/Octave for hands-on exercise. You can use [Coursera materials](ml-class.org) for the exercises.
With the passing years, machine learning implementation in python has gained traction. Exercises in Python skeleton are available [here](https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/)
### Lecture 1
Skip this unless you really don't have an idea of what is machine learning and what are the general groups of machine learning algorithms.

### Lecture 2
* Covers page 1 to 11 of Lecture Note 1
* Knowledge in Matrix operations and Vector Calulus is needed
* The normal equations part could be generalized with pseudoinverse inverse matrix operations, particularly, Moore-Penrose inverse.

### Lecture 3
* **Parametric** vs **Non-parametric** Learning Algorithm
  - **Parametric** - Fixed set of parameters
  - **Non-parametric** - Amount of parameters scale with training set size
* **Locally Weighted Regression (Loess/Lowess)** 
  - non-parametric
  - nearest-neighbor + linear regression (least square error function).
* **Probabilistic Interpretation of Linear Regression** 
  - Given <img src="https://latex.codecogs.com/gif.latex?y" title="y" /> as a function of <img src="https://latex.codecogs.com/gif.latex?h_{\Theta&space;}(x)" title="h_{\Theta }(x)" /> and error <img src="https://latex.codecogs.com/gif.latex?\varepsilon" title="\varepsilon" />.
  
    <p> <img src="https://latex.codecogs.com/gif.latex?y^{i}&space;=&space;\Theta&space;^{T}x^{i}&space;&plus;&space;\varepsilon&space;^{i}&space;\rightarrow&space;\varepsilon&space;^{i}&space;=&space;y^{i}&space;-&space;\Theta&space;^{T}x^{i}" title="y^{i} = \Theta ^{T}x^{i} + \varepsilon ^{i} \rightarrow \varepsilon ^{i} = y^{i} - \Theta ^{T}x^{i}" />
  - We want to model/fit/predict the value of <img src="https://latex.codecogs.com/gif.latex?y" title="y" /> given <img src="https://latex.codecogs.com/gif.latex?x" title="x" />. We want <img src="https://latex.codecogs.com/gif.latex?y" title="y" /> and <img src="https://latex.codecogs.com/gif.latex?\Theta&space;^{T}x" title="\Theta ^{T}x" /> to be as close as possible, that is, we want to minimize the error <img src="https://latex.codecogs.com/gif.latex?\varepsilon" title="\varepsilon" /> between the two terms. 
  - Assuming a Gaussian Distribution for the error (see Central Limit Theorem). 
    <p> <img src="https://latex.codecogs.com/gif.latex?p(\varepsilon&space;^{(i)})=\frac{1}{\sigma&space;\sqrt{2\pi}}e^{-\frac{(\varepsilon&space;^{(i)})^2}{2\sigma^{2}}}" title="p(\varepsilon ^{(i)})=\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{(\varepsilon ^{(i)})^2}{2\sigma^{2}}}" />
  - Substituting the first equation implies
    <p> <img src="https://latex.codecogs.com/gif.latex?p(y|x&space;;&space;\Theta)=\frac{1}{\sigma&space;\sqrt{2\pi}}e^{-\frac{(y&space;^{(i)}-\Theta&space;^{T}x^{(i)})^2}{2\sigma^{2}}}" title="p(y|x ; \Theta)=\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{(y ^{(i)}-\Theta ^{T}x^{(i)})^2}{2\sigma^{2}}}" />
    
  - the function above can be viewed as the probality or the likelihood of getting a correct value of <img src="https://latex.codecogs.com/gif.latex?y" title="y" /> given <img src="https://latex.codecogs.com/gif.latex?x" title="x" /> as parametrized by <img src="https://latex.codecogs.com/gif.latex?\Theta" title="\Theta" />.  We therefore want to maximize this "likelihood". Using MLE(Maximum Likelihood Estimation) and knowing that the Gaussian is a smooth surface, we therefore just maximize its log(or ln in other reference, read about log MLE) since log follows the trend of a function's gradient. Maximizing the likelihood function gives as 
    <p> <img src="https://latex.codecogs.com/gif.latex?constant&space;-&space;\frac{1}{2\sigma&space;^{2}}\sum&space;(y^{(i))}-\Theta&space;^{T}x^{(i))})^{2}" title="constant - \frac{1}{2\sigma ^{2}}\sum (y^{(i))}-\Theta ^{T}x^{(i))})^{2}" />
  - Maximizing the above equation is just equal to minimizing the least square function which was defined for the error/cost function of a linear regression    
  
### Lecture 4
Knowledge acquisition in progress.

## Links
* Course Page: [Stanford](http://cs229.stanford.edu/) | [SEE](https://see.stanford.edu/course/cs229)
* Lecture Videos: [Youtube](https://www.youtube.com/watch?v=UzxYlbK2c7E&list=PLA89DCFA6ADACE599) | [SEE](https://see.stanford.edu/course/cs229) 
* Couse Materials: [Excluding Solutions](https://github.com/econti/cs229) | [Including Answers and Solutions](https://see.stanford.edu/materials/aimlcs229/MachineLearningAllMaterials.zip)
* Summary of the Whole Course: [desmond-ong's CS229 Summary](https://github.com/desmond-ong/MLSummary)
* Programming Exercise: [Matlab/Octave](ml-class.org) | [Python](https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/) | [Another Python](https://github.com/kaleko/CourseraML)

## Supplementary Materials
* Visualization of Backpropagation: [Link](https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/)
* Table of Basic Derivatives and Integrals: [PDF](https://math.boisestate.edu/~shariultman/teaching/basic_derivatives_&_integrals_II.pdf)
