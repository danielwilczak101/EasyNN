import numpy as np

# This function performs an operation with two tensors. The two tensors must have
# the same dimensions.
#
# The accepted operations are:
# "a" = t1 + t2 (add)
# "s" = t1 - t2 (subtract)
# "m" = t1 * t2 (multiply)
# "d" = t1 / t2 (divide)
def arth_tensor(t1, t2, oper):

    # Check to make sure both tensors are of the same dimensions.
    t1_dim = np.shape(t1)
    t2_dim = np.shape(t2)

    try:
        if t1_dim != t2_dim:
            1/0
    except ZeroDivisionError:
        print("Dimensions are different!")
        raise

    # Perform the desired operation in the tensors.
    result = perf_op(t1, t2, oper)
    return result

# end arth_tensor

# This function makes the result from the inner-most dimension of t1 and t2
# to the outer-most dimension of t1 and t2.
def perf_op(t1, t2, oper):

    # If we are at the inner-most dimension, perform the desired
    # operation and store it in "base". After we performed the operation
    # on the entire array, we return "base".
    if len(np.shape(t1)) == 1:

        # This represents the last dimension.
        base = []*len(t1)

        # Perform the operation for each element.
        for index in range(len(t1)):

            if oper == "a":
                base.append(t1[index] + t2[index])
            elif oper == "s":
                base.append(t1[index] - t2[index])
            elif oper == "m":
                base.append(t1[index] * t2[index])
            elif oper == "d":
                base.append(t1[index] / t2[index])
            else:

                try:
                    1/0
                except ZeroDivisionError:
                    print("Invalid operation!")
                    raise

        # We return here. If we didn't, we would recurse again.
        # If we recurse again, the t1 and t2 parameters for this
        # function would be single variables (no longer arrays).
        return base

    # This is what will be the final answer (depending what dimension we
    # are currently on). "result" will grow from the inside-out.
    result = []*len(t1)

    # For each dimension, add the data from inner-dimensions.
    for index in range(len(t1)):

        # Get the data from the inner-dimensions of t1 and t2.
        inner_t1 = t1[index]
        inner_t2 = t2[index]

        # After the operation is performed on the inner-dimensions of
        # t1 and t2, we append it to "result".
        result.append(perf_op(inner_t1, inner_t2, oper))

    return result

# end operate

def main():

    t1 = [[3, 7], [1, 9], [90, 900]]
    t2 = [[4, 90], [8, 89], [3000, -5000]]

    t3 = arth_tensor(t1, t2, "a")
    print(t1)
    print(t2)
    print(t3)

main()