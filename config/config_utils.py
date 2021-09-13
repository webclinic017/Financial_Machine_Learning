
from collections import defaultdict
import logging
import os
from xml.etree import ElementTree as ET

LOGGER = logging.getLogger(__name__)


class ConfigUtils:
    """Basic configuration file utilities class."""

    def __init__(self, folder: str = "utils",
                 filename: str = "config.xml") -> None:
        """Initialise ConfigUtils class."""
        self.folder = folder
        self.filename = filename
        self.cfg_dict = None

        self.cfg_avail()
        self.convert_cfg()

    @property
    def cfg_path(self) -> str:
        """
        Return path of configuration file.

        As mentioned in the module description before, the
        configuration file must be located in the same folder as
        config_utils.py.

        Returns
        -------
        str
            Configuration file path.

        """
        if self.folder == "tests":
            return os.path.abspath(os.path.join(self.folder, self.filename))
        else:
            return os.path.abspath(os.path.join(__file__, os.pardir,
                                                self.filename))

    def cfg_avail(self) -> None:
        """
        Check if configuration file is available.

        Raises
        ------
        ConfigUtilsError
            Custom error if config file is not present.

        """
        if os.path.isfile(self.cfg_path):
            LOGGER.info("Configuration file available")
        else:
            raise ConfigFileNotFoundError()

    def __etree_to_dict(self, t):
        """Convert root of config xml recursively to dictionary."""
        d = {t.tag: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(self.__etree_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {t.tag: {k: v[0] if len(v) == 1 else v
                         for k, v in dd.items()}}
        if t.attrib:
            d[t.tag].update((k, v) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[t.tag]['#text'] = text
            else:
                d[t.tag] = text
        return d

    def convert_cfg(self) -> None:
        """Store config xml file in json-like dictionary."""
        with open(self.cfg_path, "rt") as config:
            tree = ET.parse(config)
            root = tree.getroot()
            cfg_dict = self.__etree_to_dict(root)
            self.cfg_dict = cfg_dict

    def read_cfgvalue(self, section: str, c_param: str) -> str:
        """
        Read configuration parameter from section in config file.

        Raises custom errors if section or parameter not available.

        Parameters
        ----------
        section : str
            Section in the config file (e.g. 'Central_DB').
        c_param : str
            Configuration parameter in defined section (e.g. 'host').

        Returns
        -------
        str
            Value of config parameter (e.g. 'localhost').

        """
        root = self.cfg_dict["ConditionMonitoring"]["initID"]
        sections = root.keys()
        if section in sections:
            c_params = root[section].keys()
            if c_param in c_params:
                cfg_value = root[section][c_param]
                if not cfg_value:
                    raise ConfigParamValueError(c_param, section)
            else:
                raise ConfigParamNotFoundError(c_param, section)
        else:
            raise ConfigSectionNotFoundError(section)
        return cfg_value

    def read_cfgvalue_old(self, section: str, c_param: str) -> str:
        """
        Read config file and configuration parameter in section.

        ----> DEPRECATED method <-----

        Parameters
        ----------
        section : str
            Section in the config file (e.g. 'Central_DB').
        c_param : str
            Configuration parameter in defined section (e.g. 'host').

        Returns
        -------
        str
            Value of config parameter (e.g. 'localhost').

        """
        exception = (f"Could not read config parameter '{c_param}' from "
                     + f"section '{section}' in config file")
        with open(self.cfg_path, "rt") as config:
            tree = ET.parse(config)
            if list(tree.iter(section)):
                for node in tree.iter(section):
                    if c_param in node.attrib.keys():
                        c_value = node.attrib.get(c_param)
                    else:
                        # Config file section OK, parameter wrong
                        LOGGER.error(exception)
                        raise ConfigParamValueError(exception=exception)
            else:
                # Config file section wrong
                LOGGER.error(exception)
                raise ConfigSectionNotFoundError(exception=exception)
        return c_value


class ConfigUtilsGeneralError(Exception):
    """Parent custom class raised for general errors in config."""

    def __init__(self, exception,
                 message="Error handling configuration file: "):
        """Initialise of custom ConfigUtilsError."""
        self.message = f"{message}{exception}"
        super().__init__(self.message)


class ConfigSectionNotFoundError(ConfigUtilsGeneralError):
    """Custom class raised if section not found in config file."""

    def __init__(self, section):
        """Initialise of custom ConfigUtilsError."""
        self.message = (f"Could not find section '{section}' in configuration "
                        + "file")
        super().__init__(self.message)


class ConfigParamNotFoundError(ConfigUtilsGeneralError):
    """Custom class raised if param not found in config file section."""

    def __init__(self, c_param, section):
        """Initialise of custom ConfigUtilsError."""
        self.message = (f"Could not find parameter '{c_param}' in section "
                        + f"'{section}' of configuration file")
        super().__init__(self.message)


class ConfigParamValueError(ConfigUtilsGeneralError):
    """Custom class raised if parameter empty in config file section."""

    def __init__(self, c_param, section):
        """Initialise of custom ConfigUtilsError."""
        self.message = (f"Parameter '{c_param}' in section '{section}' of "
                        + "configuration file has no value")
        super().__init__(self.message)


class ConfigFileNotFoundError(ConfigUtilsGeneralError):
    """Custom class raised if config file not found."""

    def __init__(self):
        """Initialise of custom ConfigUtilsError."""
        self.message = "Could not find config file"
        super().__init__(self.message)
