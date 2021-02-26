# This function makes a multi-dimensional array where
# in_data contains the dimensions and initialization value.
#
# The first in_data-1 parameters will be the dimensions of the matrix.
# The last parameter will be the initialization value.
# There must be at least 2 parameters.
def ini_tensor(*in_data):

    # There must be at least two parameters!
    try:
        if len(in_data) < 2:
            1/0
    except ZeroDivisionError:
        print("ini_tensor function must have at least two parameters!")
        raise

    # dim contains the dimension sizes.
    # Only the first in_data-1 parameters define the dimensions of the
    # multi-dimensional array.
    dim = []*(len(in_data) - 1)
    for dimIndex in range(len(in_data) - 1):
        dim.append(in_data[dimIndex])

    # ini_val contains the initialization value (the last parameter in in_data).
    ini_val = in_data[len(in_data) - 1]

    # To make the multi-dimensional array, we will work backwards from
    # the inner-most array to the outer-most array.

    # Reverse the order of dimensions.
    dim.reverse()
    # curr will be the current array/dimension we are on.
    # Before we iterate, we initialize it as the inner-most array.
    curr = [ini_val]*dim[0]
    # We just initialized the inner-most array, we can now skip the first element
    # in dim because we already did the tasks needed for that element.
    dim.pop(0)
    # We will now be working our way from the inner-most array to the outer-most
    # array by adding a specific number of previous arrays to the current array.
    for dim_arr in dim:
        # Store the current array in a temporary array because we
        # need to re-use the current array to store a specific number of itself.
        temp = curr.copy()
        # Remove everything in curr because it will now be storing arrays
        # of a dimension different than what curr has.
        curr.clear()
        # Add curr to itself a specific number of times.
        # (The number of times represents the current dimension).
        for amount in range(dim_arr):
            curr.append(temp)

    # We are now done initializing our multi-dimensional array.
    return curr

#end ini_tensor(*in_data)


def main():

    tensor = ini_tensor(6, 7)
    print(tensor)
main()