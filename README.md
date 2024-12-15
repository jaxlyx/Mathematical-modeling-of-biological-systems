# single neuron dynamics
  Here let's simulate a dynamic model of a single neuron. First, we wrote simulation methods for three widely used models which are Hodgekin-Huxley model, LIF model and Izhikevich model. 

  Here the HH model involves four microsquare equations, which are so complex that it becomes very impractical to use them for large-scale neural network simulations. The LIF model, on the other hand, involves only a microsquare equation, but it is so simple that it is biased for many experimental data. The Izhikevich model is a compromise between the two models, it only involves two microsquare equations consisting of membrane potential and membrane recovery parameters, and the output image is very similar to the experimental recorded neural potential, so we will use this model for further neural network simulation.
