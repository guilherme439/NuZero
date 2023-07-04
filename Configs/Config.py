import configparser
import ast 
from numbers import Number

class Config():

    def __init__():
        return

    def convert_to_string(self, value):
        if isinstance(value, str):
            # configparser converts all objects to strings internally.
            # So we need to add quotes to distinguish the ones that are actually strings.
            string = "'" + value + "'"
            
        elif isinstance(value, list):
            # Convert each element of the list
            string = "["
            for i in range(len(value)):
                string += self.convert_to_string(value[i])
                if i != (len(value)-1):
                    string += ", "
            string += "]"

        elif isinstance(value, bool):
            # Make sure booleans are not treated as integers
            string = str(value)

        elif isinstance(value, Number):
            # Numbers with more that 5 digits get converted to scientific notation
            string = format(value, '<.5g')  

        else:
            string = str(value)
        
        return string

    def save(self, filepath):
        config = configparser.ConfigParser()
        attributes = self.__dict__.items()
        for name, dict in attributes:
            if not(name.startswith('__') and name.endswith('__')):
                config[name] = {} # start the section
                for key, value in dict.items():
                    config[name][key] = self.convert_to_string(value)
        
        with open(filepath, 'w') as configfile:
            config.write(configfile)
        
        print("\nSaved config at: " + filepath)
		
    def load(self, filepath):
        config = configparser.ConfigParser()
        config.read(filepath)
        attributes = self.__dict__.items()

        for name, _ in attributes:
            if not(name.startswith('__') and name.endswith('__')):
                attr_dict = dict(config.items(name))
                for key, value in attr_dict.items():
                    attr_dict[key] = ast.literal_eval(value)
                    # Since configparser stores everything as strings
                    # we use ast.literal_value to get the actual python objects.

                setattr(self, name, attr_dict)

        print("\nLoaded config from: " + filepath)
