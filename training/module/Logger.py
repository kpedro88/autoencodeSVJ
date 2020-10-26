class Logger:
    """
    good logging parent class.
    Usage:
        When creating a class, subclass this class.
        You can override the _LOG_PREFIX, _ERROR_PREFIX,
        and VERBOSE member variables either in the derived
        class __init__, or when calling the logger __init__.

        Ex:
        class <classname>(logger):
            def __init__(self, ..., verbose):
                ...
                logger.__init__(self, VERBOSE=verbose, LOG_PREFIX="MY class log prefix")
                ...
            OR
            def __init__(self, ..., verbose):
                logger.__init__(self)
                self.VERBOSE = verbose
                self._LOG_PREFIX = "my log prefix: "

        From there, you can use the logging functions as members of your class.
        i.e. self.log("log message") or self.error("log message")

        To get the log as a string, pass the 'string=True' argument to
        either of the 'log' or 'error' member functions

    """
    
    def __init__(self, LOG_PREFIX="logger :: ", ERROR_PREFIX="ERROR: ", VERBOSE=True):
        self._LOG_PREFIX = LOG_PREFIX
        self._ERROR_PREFIX = ERROR_PREFIX
        self.VERBOSE = VERBOSE
    
    def log(self, s, string=False):
        if string:
            if self.VERBOSE:
                return self._log_str(s, self._LOG_PREFIX)
            return ''
        if self.VERBOSE:
            self._log_base(s, self._LOG_PREFIX)
    
    def error(self, s, string=False):
        if string:
            return self._log_str(s, self._LOG_PREFIX + self._ERROR_PREFIX)
        self._log_base(s, self._LOG_PREFIX + self._ERROR_PREFIX)
    
    def _log_base(self, s, prefix):
        if isinstance(s, str):
            for line in s.split('\n'):
                print((prefix + str(line)))
        else:
            print((prefix + str(s)))
    
    def _log_str(self, s, prefix):
        out = ''
        if isinstance(s, str):
            for line in s.split('\n'):
                out += prefix + str(line) + '\n'
        else:
            out += prefix + str(line) + '\n'
        return out