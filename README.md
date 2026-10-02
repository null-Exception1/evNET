# progress report log 1

- simple synapses and neurons added (synapses with delays, neurons with ffds)
- plans to add homeostasis

https://github.com/user-attachments/assets/39a48cb9-949f-43b7-8f3c-9b3dc753690e

# progress report log 2

- added homeostasis so deadlocks and recurrent loops will calm down
- plans to add fine tuning NEAT - randomize wiring and synapse weights

https://github.com/user-attachments/assets/b3acce7a-5bcd-454a-9e35-094e832bb35d

# progress report log 3

- simple NEAT rewarding singular route from input to output neuron with one fire

https://github.com/user-attachments/assets/15c9c175-6b4c-4ba9-8205-f3940d6dd08f

# progress report log 4

- scaled up fine tuning and fixed some bugs
- this task required each input neuron at their respective order to be routed to output neuron in that order
- introduced modularity via occam's razor technique

https://github.com/user-attachments/assets/f86d7b4c-cb65-4708-838b-6301881d792a

# progress report log 5

- outerNEAT added (NEAT deciding brain's neuron positions)
- randomizing neuron positions
- added chemical volume transmission
- complex gating of neurons as ffd networks with the introduction of chemicals confirmed, gating is controlked purely by the weights and biases of the ffd, which has proven itself to have complex mechanisms by thorough playground testing. i believe neat can take advantage of this to form complex gates.
- misfire bug which causes the 2nd input neuron to not be able to recreate the same conditions causing 1st output neuron to fire later after 2nd output neuron firing
  (still debugging, causes remain unknown) but scored perfect in an isolated test somehow
- confirmed NEAT can treat chemicals one of the fundamental blocks to build brain
- save features added
- visual playground added
  
https://github.com/user-attachments/assets/6be12b20-287e-4437-b353-f18433e49876

# TODO 

- will add it so that instead of from scratch random creatures, i mutate 20% of the neurons to be positionally different or of different types
- should cut back on the general noise allowing for more refined structures
- mutation parameters that adapts to the training phases would be interesting
- need to add a more complex task to solve, more neurons in a larger space
- or more possibly just more of the same but longer generations (more than 5 previously)
- try to recreate functions like xor, and etc.

- sooner than later will implement an active AGENT focused task that will involve a physical space around which the model can move and get rewarded for
- need to add a cuda kernel that will speed up NEAT
- eliminate redundant parts of the scripts that take up processing power



