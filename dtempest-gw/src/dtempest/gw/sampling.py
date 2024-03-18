from functools import partialmethod

from dtempest.core.sampling import SampleDict, SampleSet, MSEDataFrame, MSESeries
from .config import cbc_jargon


class CBCMSESeries(MSESeries):
    __init__ = partialmethod(MSESeries.__init__, jargon=cbc_jargon)


class CBCMSEDataFrame(MSEDataFrame):
    __init__ = partialmethod(MSEDataFrame.__init__, jargon=cbc_jargon, _series_class=CBCMSESeries)


class CBCSampleDict(SampleDict):
    __init__ = partialmethod(SampleDict.__init__, jargon=cbc_jargon, _series_class=CBCMSESeries)

    @property
    def plotting_map(self):
        existing = super(CBCSampleDict, self).plotting_map
        modified = existing.copy()
        modified.update(
            {
                "marginalized_posterior": self._marginalized_posterior,
                # "skymap": self._skymap,
                "hist": self._marginalized_posterior,
                "corner": self._corner,
                # "spin_disk": self._spin_disk,
                "2d_kde": self._2d_kde,
                "triangle": self._triangle,
                "reverse_triangle": self._reverse_triangle,
            }
        )
        return modified

    def generate_all_posterior_samples(self, function=None, **kwargs):
        """Convert samples stored in the SamplesDict according to a conversion
        function

        Parameters
        ----------
        function: func, optional
            function to use when converting posterior samples. Must take a
            dictionary as input and return a dictionary of converted posterior
            samples. Default `pesummary.gw.conversions.convert
        **kwargs: dict, optional
            All additional kwargs passed to function
        """
        if function is None:
            from pesummary.gw.conversions import convert
            function = convert
        _samples = self.copy()
        _keys = list(_samples.keys())
        kwargs.update({"return_dict": True})
        out = function(_samples, **kwargs)
        if kwargs.get("return_kwargs", False):
            converted_samples, extra_kwargs = out
        else:
            converted_samples, extra_kwargs = out, None
        for key, item in converted_samples.items():
            if key not in _keys:
                self[key] = item
        return extra_kwargs

    def _skymap(self, **kwargs):
        """Wrapper for the `pesummary.gw.plots.plot._ligo_skymap_plot`
        function

        Parameters
        ----------
        **kwargs: dict
            All kwargs are passed to the `_ligo_skymap_plot` function
        """
        from pesummary.gw.plots.plot import _ligo_skymap_plot

        if "luminosity_distance" in self.keys():
            dist = self["luminosity_distance"]
        else:
            dist = None

        return _ligo_skymap_plot(self["ra"], self["dec"], dist=dist, **kwargs)

    def _spin_disk(self, **kwargs):
        """Wrapper for the `pesummary.gw.plots.publication.spin_distribution_plots`
        function
        """
        from pesummary.gw.plots.publication import spin_distribution_plots

        required = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        if not all(param in self.keys() for param in required):
            raise ValueError(
                "The spin disk plot requires samples for the following "
                "parameters: {}".format(", ".join(required))
            )
        samples = [self[param] for param in required]
        return spin_distribution_plots(required, samples, None, **kwargs)

    def _2d_kde(self, parameters: list, module="gw", **kwargs):
        """Wrapper for the `pesummary.gw.plots.publication.twod_contour_plot` or
        `pesummary.core.plots.publication.twod_contour_plot` function

        Parameters
        ----------
        parameters: list
            list of length 2 giving the parameters you wish to plot
        module: str, optional
            module you wish to use for the plotting
        **kwargs: dict, optional
            all additional kwargs are passed to the `twod_contour_plot` function
        """
        from pesummary.gw.plots.publication import twod_contour_plots
        if module == "gw":
            return twod_contour_plots(
                parameters, [[self[parameters[0]], self[parameters[1]]]],
                [None], {
                    parameters[0]: self.latex_labels[parameters[0]],
                    parameters[1]: self.latex_labels[parameters[1]]
                }, **kwargs
            )
        else:
            raise NotImplementedError
        # return getattr(_module, "twod_contour_plot")(
        #     self[parameters[0]], self[parameters[1]],
        #     xlabel=self.latex_labels[parameters[0]],
        #     ylabel=self.latex_labels[parameters[1]], **kwargs
        # )

    def _triangle(self, parameters: list, module="gw", **kwargs):
        """Wrapper for the `pesummary.core.plots.publication.triangle_plot`
        function

        Parameters
        ----------
        parameters: list
            list of parameters they wish to study
        **kwargs: dict
            all additional kwargs are passed to the `triangle_plot` function
        """
        from pesummary.gw.plots.publication import triangle_plot
        if module == "gw":
            kwargs["parameters"] = parameters
        else:
            raise NotImplementedError
        return triangle_plot(
            (self[parameters[0]]), (self[parameters[1]]),
            xlabel=self.latex_labels[parameters[0]],
            ylabel=self.latex_labels[parameters[1]], **kwargs
        )


class CBCSampleSet(SampleSet):
    __init__ = partialmethod(SampleSet.__init__,
                             _series_class=CBCMSESeries, _dataframe_class=CBCMSEDataFrame, jargon=cbc_jargon)
