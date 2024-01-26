from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")
print(tf.__version__) # find the version number (should be 2.x+)

print("")
print("# Create a scalar (rank 0 tensor)")
scalar = tf.constant(7)
print(scalar)
print(scalar.ndim)

print("")
print("# Create a vector (more than 0 dimensions)")
vector = tf.constant([10, 10])
print(vector)
print(vector.ndim)

print("")
print("# Create a matrix (more than 1 dimension)")
matrix = tf.constant([[10, 7],
                      [7, 10]])
print(matrix)
print(matrix.ndim)

print("")
print("# Create another matrix and define the datatype")
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16) # specify the datatype with 'dtype'
print(another_matrix)
print("# Even though another_matrix contains more numbers, its dimensions stay the same")
print(another_matrix.ndim)

print("")
print("# How about a tensor? (more than 2 dimensions, although, all of the above items are also technically tensors)")
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
print(tensor)
print(tensor.ndim)

print("")
print("# Create the same tensor with tf.Variable() and tf.constant()")
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
print(changeable_tensor)
print(unchangeable_tensor)

# Will error (requires the .assign() method)
#changeable_tensor[0] = 7
#print(changeable_tensor)

print("")
print("# Won't error")
changeable_tensor[0].assign(7)
print(changeable_tensor)

print("")
print("# Create two random (but the same) tensors")
random_1 = tf.random.Generator.from_seed(42) # set the seed for reproducibility
random_1 = random_1.normal(shape=(3, 2)) # create tensor from a normal distribution
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))

print("")
print("# Are they equal?")
print(random_1, random_2, random_1 == random_2)

print("")
print("# Create two random (and different) tensors")
random_3 = tf.random.Generator.from_seed(42)
random_3 = random_3.normal(shape=(3, 2))
random_4 = tf.random.Generator.from_seed(11)
random_4 = random_4.normal(shape=(3, 2))

print("# Check the tensors and see if they are equal")
print(random_3, random_4, random_1 == random_3, random_3 == random_4)

print("")
print("# Shuffle a tensor (valuable for when you want to shuffle your data)")
not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2, 5]])
print("# Gets different results each time")
print(tf.random.shuffle(not_shuffled))

print("")
print("# Shuffle in the same order every time using the seed parameter (won't acutally be the same)")
print(tf.random.shuffle(not_shuffled, seed=42))

print("# Shuffle in the same order every time")

print("# Set the operation random seed")
print(tf.random.shuffle(not_shuffled, seed=42))

print("")
print("# Set the operation random seed")
print(tf.random.shuffle(not_shuffled))

print("")
print("# Make a tensor of all ones")
print(tf.ones(shape=(3, 2)))

print("")
print("# Make a tensor of all zeros")
print(tf.zeros(shape=(3, 2)))

print("")
print("# Create a NumPy array between 1 and 25")
numpy_A = np.arange(1, 25, dtype=np.int32)
print("# Note: the shape total (2*4*3) has to match the number of elements in the array")
A = tf.constant(numpy_A,
                shape=[2, 4, 3])
print(numpy_A, A)

print("")
print("# Create a rank 4 tensor (4 dimensions)")
rank_4_tensor = tf.zeros([2, 3, 4, 5])
print(rank_4_tensor)

print(rank_4_tensor.shape, rank_4_tensor.ndim, tf.size(rank_4_tensor))

print("")
print("# Get various attributes of tensor")
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (2*3*4*5):", tf.size(rank_4_tensor).numpy()) # .numpy() converts to NumPy array

print("")
print("# Get the first 2 items of each dimension")
print(rank_4_tensor[:2, :2, :2, :2])

print("")
print("# Get the dimension from each index except for the final one")
print(rank_4_tensor[:1, :1, :1, :])

print("")
print("# Create a rank 2 tensor (2 dimensions)")
rank_2_tensor = tf.constant([[10, 7],
                             [3, 4]])
print("# Get the last item of each row")
print(rank_2_tensor[:, -1])

print("")
print("# Add an extra dimension (to the end)")
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # in Python "..." means "all dimensions prior to"
print(rank_2_tensor, rank_3_tensor) # shape (2, 2), shape (2, 2, 1)

print("")
print("# -1 means last axis")
print(tf.expand_dims(rank_2_tensor, axis=-1))

print("")
print("# You can add values to a tensor using the addition operator")
tensor = tf.constant([[10, 7], [3, 4]])
print(tensor + 10)

print("")
print("# Original tensor unchanged")
print(tensor)

print("")
print("# Multiplication (known as element-wise multiplication)")
print(tensor * 10)

print("")
print("# Subtraction")
print(tensor - 10)

print("")
print("# Use the tensorflow function equivalent of the '*' (multiply) operator")
print(tf.multiply(tensor, 10))

print("")
print("# The original tensor is still unchanged")
print(tensor)

print("")
print("# Matrix multiplication in TensorFlow")
print(tensor)
print(tf.matmul(tensor, tensor))

print("")
print("# Matrix multiplication with Python operator '@'")
print(tensor @ tensor)

print("")
print("# Create (3, 2) tensor")
X = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])

print("# Create another (3, 2) tensor")
Y = tf.constant([[7, 8],
                 [9, 10],
                 [11, 12]])
print(X, Y)

# Try to matrix multiply them (will error)
#X @ Y

print("")
print("# Example of reshape (3, 2) -> (2, 3)")
print(tf.reshape(Y, shape=(2, 3)))

print("")
print("# Try matrix multiplication with reshaped Y")
print(X @ tf.reshape(Y, shape=(2, 3)))

print("")
print("# Example of transpose (3, 2) -> (2, 3)")
tf.transpose(X)

print("# Try matrix multiplication")
print(tf.matmul(tf.transpose(X), Y))

print("")
print("# You can achieve the same result with parameters")
print(tf.matmul(a=X, b=Y, transpose_a=True, transpose_b=False))

print("")
print("# Perform the dot product on X and Y (requires X to be transposed)")
print(tf.tensordot(tf.transpose(X), Y, axes=1))

print("")
print("# Perform matrix multiplication between X and Y (transposed)")
print(tf.transpose(Y))
print(tf.matmul(X, tf.transpose(Y)))

print("")
print("# Perform matrix multiplication between X and Y (reshaped)")
print(tf.matmul(X, tf.reshape(Y, (2, 3))))

print("")
print("# Check shapes of Y, reshaped Y and tranposed Y")
print(Y.shape, tf.reshape(Y, (2, 3)).shape, tf.transpose(Y).shape)

print("")
print("# Check values of Y, reshape Y and tranposed Y")
print("Normal Y:")
print(Y, "\n") # "\n" for newline

print("Y reshaped to (2, 3):")
print(tf.reshape(Y, (2, 3)), "\n")

print("Y transposed:")
print(tf.transpose(Y))

print("")
print("# Create a new tensor with default datatype (float32)")
B = tf.constant([1.7, 7.4])

print("# Create a new tensor with default datatype (int32)")
C = tf.constant([1, 7])
print(B, C)

print("")
print("# Change from float32 to float16 (reduced precision)")
B = tf.cast(B, dtype=tf.float16)
print(B)

print("")
print("# Change from int32 to float32")
C = tf.cast(C, dtype=tf.float32)
print(C)

print("")
print("# Create tensor with negative values")
D = tf.constant([-7, -10])
print(D)

print("")
print("# Get the absolute values")
print(tf.abs(D))

print("")
print("# Create a tensor with 50 random values between 0 and 100")
E = tf.constant(np.random.randint(low=0, high=100, size=50))
print(E)

print("")
print("# Find the minimum")
print(tf.reduce_min(E))

print("")
print("# Find the maximum")
print(tf.reduce_max(E))

print("")
print("# Find the mean")
print(tf.reduce_mean(E))

print("")
print("# Find the sum")
print(tf.reduce_sum(E))

print("")
print("# Create a tensor with 50 values between 0 and 1")
F = tf.constant(np.random.random(50))
print(F)

print("")
print("# Find the maximum element position of F")
print(tf.argmax(F))

print("")
print("# Find the minimum element position of F")
print(tf.argmin(F))

print("")
print("# Find the maximum element position of F")
print(f"The maximum value of F is at position: {tf.argmax(F).numpy()}")
print(f"The maximum value of F is: {tf.reduce_max(F).numpy()}")
print(f"Using tf.argmax() to index F, the maximum value of F is: {F[tf.argmax(F)].numpy()}")
print(f"Are the two max values the same (they should be)? {F[tf.argmax(F)].numpy() == tf.reduce_max(F).numpy()}")

print("")
print("# Create a rank 5 (5 dimensions) tensor of 50 numbers between 0 and 100")
G = tf.constant(np.random.randint(0, 100, 50), shape=(1, 1, 1, 1, 50))
print(G.shape, G.ndim)

print("")
print("# Squeeze tensor G (remove all 1 dimensions)")
G_squeezed = tf.squeeze(G)
print(G_squeezed.shape, G_squeezed.ndim)

print("")
print("# Create a list of indices")
some_list = [0, 1, 2, 3]

print("# One hot encode them")
print(tf.one_hot(some_list, depth=4))

print("")
print("# Specify custom values for on and off encoding")
print(tf.one_hot(some_list, depth=4, on_value="We're live!", off_value="Offline"))

print("")
print("# Create a new tensor")
H = tf.constant(np.arange(1, 10))
print(H)

print("")
print("# Square it")
print(tf.square(H))

# Find the squareroot (will error), needs to be non-integer
#tf.sqrt(H)

print("")
print("# Change H to float32")
H = tf.cast(H, dtype=tf.float32)
print(H)

print("")
print("# Find the square root")
print(tf.sqrt(H))

print("")
print("# Find the log (input also needs to be float)")
print(tf.math.log(H))

print("")
print("# Create a variable tensor")
I = tf.Variable(np.arange(0, 5))
print(I)

print("")
print("# Assign the final value a new value of 50")
print(I.assign([0, 1, 2, 3, 50]))

print("")
print("# The change happens in place (the last value is now 50, not 4)")
print(I)

print("")
print("# Add 10 to every element in I")
print(I.assign_add([10, 10, 10, 10, 10]))

print("")
print("# Again, the change happens in place")
print(I)

print("")
print("# Create a tensor from a NumPy array")
J = tf.constant(np.array([3., 7., 10.]))
print(J)

print("")
print("# Convert tensor J to NumPy with np.array()")
print(np.array(J), type(np.array(J)))

print("")
print("# Convert tensor J to NumPy with .numpy()")
print(J.numpy(), type(J.numpy()))

print("")
print("# Create a tensor from NumPy and from an array")
numpy_J = tf.constant(np.array([3., 7., 10.])) # will be float64 (due to NumPy)
tensor_J = tf.constant([3., 7., 10.]) # will be float32 (due to being TensorFlow default)
print(numpy_J.dtype, tensor_J.dtype)

print("")
print("# Create a simple function")
def function(x, y):
  return x ** 2 + y

x = tf.constant(np.arange(0, 10))
y = tf.constant(np.arange(10, 20))
print(function(x, y))

print("")
print("# Create the same function and decorate it with tf.function")
@tf.function
def tf_function(x, y):
  return x ** 2 + y

print(tf_function(x, y))

print("")
print("# Finding access to GPUs")
print(tf.config.list_physical_devices('GPU'))

#print(subprocess.run(['nvidia-smi']))
