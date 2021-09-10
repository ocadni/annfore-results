import logging

def init_log(file_p):
    logging.basicConfig(filename=file_p,format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')



def attach_logger_file(logger,filename):
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    #return logger

def format_time(deltat, debug=False, compact=False):
    """
    Deltat in seconds
    """
    string_t = ""
    div = 3600
    interv_in_secs = [24*3600, 3600, 60]
    if deltat < 1:
        m = int(deltat*1000)
        return "{:3d} ms".format(m)
    mcount = deltat
    rest=deltat
    if compact:
        intv_arrs = ["d","h","m"]
        placesh = ("{:2d}", " ")
        secs_txt="s"
    else:
        intv_arrs = ["days","h","min"]
        placesh = ("{:2d} ", ", ")
        secs_txt = "secs"
    for intv, secs in zip(intv_arrs,interv_in_secs):
        if mcount / secs >= 1:
            nint = int(mcount/secs)
            rest = mcount % secs
            if debug:
                print(nint,rest)

            string_t += placesh[0].format(nint) + intv+ placesh[1]

            mcount = rest
    string_t += "{:2d} {}".format(int(rest), secs_txt)
    if deltat < 30:
        rest -= int(rest)
        string_t+=", {:3d} ms".format(int(rest*1000))
    return string_t
    #print(string_t,end=end)

def clear_log_handlers(logger):
    handlers =  list(logger.handlers)
    for l in handlers:
        logger.removeHandler(l)