import time


def timer(func: callable):

    def inner(*args, **kwargs):
        """ Decorated function """

        """ calculate time """
        start = time.perf_counter()
        answer = func(*args, **kwargs)
        end = time.perf_counter() - start

        """ print time """
        print(f'{func.__name__}() function processed in {end:0.4f} seconds')

        """ return decorated function answer """
        return answer

    return inner
