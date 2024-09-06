"""Source code for the next prime number function."""


def is_prime(n: int) -> bool:
    """Check if a number is prime.

    :param n: The number to check.
    :return: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def next_prime(current_number: int) -> int:
    """Find the next prime number after the current number.

    :param current_number: The current number.
    :return: The next prime number.
    """
    next_number = current_number + 1
    while not is_prime(next_number):
        next_number += 1
    return next_number
