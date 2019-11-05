from question1.gaussian import gaussian

def update(c, n, p, m, sigma, view, v, lamb, u):
    # Update cluster centrois v
    for i in range(c):
        for j in range(p):
            a = 0
            b = 0
            for k in range(n):
                a += ((u[i,k])**m) * gaussian(view[k,j], v[i,j], sigma[j]) * view[k,j]
                b +=  ((u[i,k])**m) * gaussian(view[k,j], v[i,j], sigma[j])
            v[i,j] = a / b

    # Update features weights
    a = 1
    for j in range(p):
        b = 0
        for i in range(c):
            for k in range(n):
                b += ((u[i,k])**m) * (2 * (1 - gaussian(view[k,j], v[i,j], sigma[j])))
        a *= b
    for j in range(p):
        b = 0
        for i in range(c):
                for k in range(n):
                    b += ((u[i,k])**m) * (2 * (1 - gaussian(view[k,j], v[i,j], sigma[j])))

        lamb[j] = (a ** (1/p)) / b
        
    # Update fuzzy membership degree 
    for i in range(c):
        for k in range(n):
            a = 0
            for h in range(c):
                phi_a = 0
                phi_b = 0
                for j in range(p):
                    phi_a += lamb[j] * 2 *(1 - gaussian(view[k,j], v[i,j], sigma[j]))
                    phi_b += lamb[j] * 2 *(1 - gaussian(view[k,j], v[h,j], sigma[j]))
                a += (phi_a / phi_b)**(1/(m-1))
            u[i,k] = a ** (-1)
    
    return {'u': u, 'lamb': lamb, 'v': v}