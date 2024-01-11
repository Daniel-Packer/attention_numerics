from jax import numpy as jnp, vmap

def get_perm_matrix(permutation):
    perm_matrix = jnp.zeros((4, 4))
    for i, j in enumerate(permutation):
        perm_matrix = perm_matrix.at[i, j].set(1)
    return perm_matrix


def get_tetrahedron_pts(method):
    match method:
        case "dan":
            tetrahedron_pts = jnp.eye(4) - 1 / 4
            _eig_vals, eig_vecs = jnp.linalg.eigh(tetrahedron_pts)
            tetrahedron_pts = tetrahedron_pts @ eig_vecs[:, 1:]
        case "ben":
            tetrahedron_pts = jnp.array(
                [
                    [1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, -1],
                    [-1, 1, -1],
                ]
            ).astype(float)
    return tetrahedron_pts

tetrahedron_pts = get_tetrahedron_pts("ben")

cube_pts = jnp.array(
    [
        [[(-1) ** i, (-1) ** j, (-1) ** k] for k in range(2)]
        for i in range(2)
        for j in range(2)
    ]
).reshape(-1, 3)


def tetrahedron_represent(permutation):
    perm_matrix = get_perm_matrix(permutation)
    P_1 = tetrahedron_pts
    P_2 = perm_matrix @ P_1
    A = jnp.linalg.pinv(P_1) @ P_2
    return A


def cube_represent(permutation):
    first_four = cube_pts[jnp.array([0, 1, 3, 2])]
    P_1 = first_four.astype(float)[:3]
    P_2 = first_four[permutation].astype(float)[:3]

    for signs in cube_pts:
        A = jnp.linalg.pinv(P_1) @ (P_2 * signs[:, None])
        if jnp.allclose(jnp.linalg.det(A), 1.0, atol=1e-5) and jnp.allclose(
            A @ A.T, jnp.eye(3), atol=1e-5
        ):
            break
    return A


def compose_permutations(perm_1, perm_2):
    return perm_1[perm_2]


S_4 = jnp.array(
    [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1], 
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ]
)

# Symmetries of a tetrahedron
U_1 = jnp.array([tetrahedron_represent(permutation) for permutation in S_4])

# Symmetries of a cube
U_2 = jnp.array([cube_represent(permutation) for permutation in S_4])


# Tests whether the representation is associative
def test_repr(represent):
    for i in range(24):
        for j in range(i, 24):
            try:
                # note that the order of the composition is "reversed"
                assert jnp.allclose(
                    represent(compose_permutations(S_4[i], S_4[j])),
                    represent(S_4[j]) @ represent(S_4[i]),
                    atol=1e-5,
                )
            except:
                print(f"Error thrown at {(i, )}")


def regular_represent(permutation):
    new_perms = vmap(compose_permutations, in_axes=(None, 0))(permutation, S_4)
    representation = jnp.all(S_4[:, None, :] == new_perms[None, :, :], axis=-1).astype(
        int
    )
    return representation.T


def check_equivariant(map, input, rep_1, rep_2):
    for i in range(24):
        assert jnp.allclose(
            map(rep_1[i] @ input), rep_2[i] @ map(input), atol=1e-5
        )
    print("Passed test!")