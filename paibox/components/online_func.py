class STDPLinear(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise NOT module. Do a bitwise NOT of the output spike of one neuron & output.

        Args:
            - neuron: the operand.

        NOTE: the inherent delay of the module is 0.
        """
        if keep_shape:
            shape_out = neuron.shape_out
        else:
            shape_out = (neuron.num_out,)

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        return x1 == 0  # x1 is an array in uint8

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        n1_not = OnlineNeuron(
            self.shape_out,
            threshold=1,
            leak_v=1,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.source[0], n1_not, -1, ConnType.One2One, name=f"s0_{self.name}"
        )

        generated = [n1_not, syn1]
        self._rebuild_out_intf(network, n1_not, *generated, **build_options)

        return generated