class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    @classmethod
    def reset_instance(cls):
        """Clear the singleton instance to allow re-initialization."""
        if cls in cls._instances:
            del cls._instances[cls]