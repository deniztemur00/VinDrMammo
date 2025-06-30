from dataclasses import dataclass



@dataclass
class GMICConfig:
    """Model specific configuration for GMIC."""
    input_channels: int = 3
