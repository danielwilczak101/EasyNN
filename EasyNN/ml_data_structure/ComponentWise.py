import numpy as np

# This function performs an operation with a tensor and a specified function.
def com_wise(fxn, tns):

    result = perf_fxn(fxn, tns)
    return result

#end com_wise

def perf_fxn(fxn, tns):

    # If we are at the inner-most dimension, perform the desired
    # function and store it in "base". After we have our result from the function
    # on the entire array, we return "base".
    if len(np.shape(tns)) == 1:

        # This represents the last dimension.
        base = [] * len(tns)

        # Use the function for each element.
        for index in range(len(tns)):

            base.append(fxn(tns[index]))

        # We return here. If we didn't, we would recurse again.
        # If we recurse again, the tns parameters for
        # perf_fxn would be single variables (no longer arrays).
        return base

    # This is what will be the final answer (depending what dimension we
    # are currently on). "result" will grow from the inside-out.
    result = [] * len(tns)

    # For each dimension, add the data from inner-dimensions.
    for index in range(len(tns)):

        # Get the data from the inner-dimensions of tns.
        inner_tns = tns[index]

        # After the function is performed on the inner-dimensions of
        # tns, we append it to "result".
        result.append(perf_fxn(fxn, inner_tns))

    return result


# end operate


def main():

    def f(x):
        return 2 + x

    t1 = [[3, 7], [1, 9], [90, 900]]

    t2 = com_wise(f, t1)
    print(t1)
    print(t2)


main()