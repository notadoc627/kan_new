from kan import KAN

def create_kan():
    return KAN(width=[7**2, 10, 10], grid=4, k=2)
model = create_kan()