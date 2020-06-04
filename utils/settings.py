def init():
    global settings_list
    settings_list = {}

def add(name, value):
    settings_list[name] = value

def get(name):
    try:
        return settings_list[name]
    except KeyError:
        print("Setting {:s} not defined, please add it to the settings first".format(str(name))) 

def replace(name, new_value):
    try:
        settings_list[name] = new_value
    except KeyError:
        print("Setting {:s} not defined, please add it to the settings first".format(str(name))) 

def delete(name):
    try:
        del settings_list[name]
    except KeyError:
        print("Setting {:s} not defined, please add it to the settings first".format(str(name))) 