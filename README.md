# TensorFlow-ML-Crash-Course
Crash Course Tensorflow ML

pre req: Creating and Manipulating Tensors
https://colab.research.google.com/notebooks/mlcc/creating_and_manipulating_tensors.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=tensors-colab&hl=en

# Exercise #2: Simulate 10 rolls of two dice.

Create a dice simulation, which generates a 10x3 2-D tensor in which:

    Columns 1 and 2 each hold one throw of one six-sided die (with values 1–6).
    Column 3 holds the sum of Columns 1 and 2 on the same row.

For example, the first row might have the following values:

    Column 1 holds 4
    Column 2 holds 3
    Column 3 holds 7

You'll need to explore the TensorFlow documentation to solve this task.

# my exercise #2 solution
import tensorflow as tf
import numpy
z = []

g = tf.Graph()
with g.as_default():
  for i in range(10):
    #print i
    
    sess = tf.InteractiveSession()

    # Dice initilization
    dice_1 = tf.Variable([1])
    dice_2 = tf.Variable([1])
    
    # solution is the variable that holds the current dice roll and sum array
    solution = tf.Variable([1])
    
    # z is the variable that holds the array inbetween loops
    if i ==0:
      z = tf.Variable([[1,1,1]])
    
    #session variable initilization
    initialization = tf.global_variables_initializer()
    sess.run(initialization) 
       
    # Rolling the dice, seed change based on loop state i
    dice1_roll = tf.random_uniform([1],1,6, dtype = tf.int32, seed = i)
    dice2_roll = tf.random_uniform([1],1,6, dtype = tf.int32, seed = i+1)
    
    #not sure all of this is required actually... might just be able
    #to put the dice1_roll directly into tf.concat below... not sure right now
    roll1=tf.to_int32(dice1_roll.eval(session=sess))
    roll2=tf.to_int32(dice2_roll.eval(session=sess))

    x = roll1.eval()
    y = roll2.eval()

    dice_1 = tf.Variable([x[0]])
    dice_2 = tf.Variable([y[0]])

    #initilizing the variables dice_1 and dice_2 that were just populated
    initialization = tf.global_variables_initializer()
    sess.run(initialization)

    #adding the values on the two die
    sum1 = tf.add(dice_1,dice_2)

    #creating a 1x3 vector representing the two die rolls and their sum
    solution = tf.concat([dice_1,dice_2,sum1],0)
    solution = tf.reshape(solution,[1,3])

        
    #need to not over write the persistant array z
    #which is going to hold the data of each trial
    if i == 0:
      z = solution  
    

    
    #need to add to persistant array z to hold all results
    if i > 0:
      z = tf.concat([solution,z],0)

    i=i+1
    
    print z.eval()
    sess.close()
 
    
# after running z.eval() ...
[[4 1 5]

 [3 4 7]
 
 [3 3 6]
 
 [5 3 8]
 
 [2 5 7]
 
 [1 2 3]
 
 [3 1 4]
 
 [2 3 5]
 
 [3 2 5]
 
 [2 3 5]]
  
