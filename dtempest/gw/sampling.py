from dtempest.core.sampling import MSESeries
from dtempest.gw.conversion import gw_jargon

# TODO: Have the jargon as model metadata and cascade from there. Hardcoded on the meantime
class GW_MSESeries(MSESeries):
    def __init__(self, name: str = None, sqrt: bool = False, jargon: dict = None, *args, **kwargs):
        super().__init__(name, sqrt, jargon=gw_jargon, *args, **kwargs)