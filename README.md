# two_headed_classifier
This is an example of a two headed classifier use case (PyTorch, Computer Vision).

In the real world tasks you often have constrains like:
- Inference speed
- Size of the model
- Exportability to mobile
- Amount of models per task (use less if possible, but without accuraccy loss)

In this case I had 2 close tasks, but one of them was more important, so it was useful to have a two-headed model.
Read more [here](https://medium.com/p/c8dc4f684091).
