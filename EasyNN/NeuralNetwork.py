class NeuralNetwork:
    """General NN structure which allows greater flexibility (and
     neuroevolution?)."""

    nodes: Sequence[NeuralNode]

    def __init__(
            self,
            nodes: Sequence[int],
            activation: Dict[int, Callable[[float], float]] = {}
    ):
        """Convert input integers to actual nodes, with activation functions
         appropriately."""

        pass


    def __call__(
            self,
            input_values: Sequence[float],
            pad: float = 0
    ) -> Sequence[float]:
        """
        Fill in node.value's with the input_values,
        fill in remaining nodes with the pad value,
        perform feed-forward propagation,
        and return the node.value's from the output_nodes.
        """

        pass


    def train(self,input_data,output_data):
        """ Train is used to take a data set and train it based on an input
        and output."""
        print("Hello NN World!")
