import numpy as np
import pytest

from core.augmentation import (
    AUGMENTATIONS,
    Compose,
    GaussianNoise,
    RandomShift,
    get_augmentation,
)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def one_pixel_images(n: int, shape: tuple[int, int], pos: tuple[int, int], bg: float = 0.0):
    """n imágenes aplanadas iguales, fondo bg y un solo píxel en 1. Forma (n, h·w)."""
    img = np.full(shape, bg)
    img[pos] = 1.0
    return np.tile(img.ravel(), (n, 1))


ALL = [
    GaussianNoise(sigma=0.1),
    GaussianNoise(sigma=0.1, clip=(0.0, 1.0)),
    RandomShift(max_px=1, image_shape=(3, 3)),
    Compose([RandomShift(max_px=1, image_shape=(3, 3)), GaussianNoise(sigma=0.1)]),
]


@pytest.mark.parametrize("aug", ALL, ids=lambda a: type(a).__name__)
def test_no_modifica_x_original(aug):
    X = rng(1).uniform(size=(6, 9))
    before = X.copy()
    out = aug(X, rng())
    np.testing.assert_array_equal(X, before)
    assert out is not X
    assert out.shape == X.shape


@pytest.mark.parametrize("aug", ALL, ids=lambda a: type(a).__name__)
def test_reproducible_con_la_misma_semilla(aug):
    X = rng(1).uniform(size=(50, 9))
    np.testing.assert_array_equal(aug(X, rng(3)), aug(X, rng(3)))
    assert not np.array_equal(aug(X, rng(3)), aug(X, rng(4)))


# --- GaussianNoise ---


def test_sigma_cero_es_identidad():
    X = rng(1).normal(size=(5, 4))
    out = GaussianNoise(sigma=0.0)(X, rng())
    np.testing.assert_array_equal(out, X)
    assert out is not X


def test_ruido_tiene_media_cero_y_desvio_sigma():
    X = np.zeros((200, 100))
    noise = GaussianNoise(sigma=0.3)(X, rng())
    assert abs(noise.mean()) < 0.01
    assert noise.std() == pytest.approx(0.3, rel=0.02)


def test_clip_escalar_y_por_feature():
    X = np.full((100, 2), 0.5)
    out = GaussianNoise(sigma=5.0, clip=(0.0, 1.0))(X, rng())
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Rango por feature, como el de una normalización min-max por columna.
    lo, hi = np.array([0.0, -2.0]), np.array([1.0, 3.0])
    out = GaussianNoise(sigma=5.0, clip=(lo, hi))(X, rng())
    assert np.all(out >= lo) and np.all(out <= hi)
    assert out[:, 1].min() < 0.0  # la segunda columna sí puede bajar de 0


@pytest.mark.parametrize("kwargs", [{"sigma": -0.1}, {"sigma": 0.1, "clip": (1.0, 0.0)}])
def test_gaussian_noise_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        GaussianNoise(**kwargs)


# --- RandomShift ---


def test_shift_mueve_un_pixel_a_una_posicion_valida():
    shape, center = (5, 5), (2, 2)
    X = one_pixel_images(300, shape, center)
    out = RandomShift(max_px=1, image_shape=shape)(X, rng())

    imgs = out.reshape(-1, *shape)
    # Cada imagen sigue teniendo exactamente un píxel encendido y el resto fondo.
    assert np.all((imgs == 1.0).sum(axis=(1, 2)) == 1)
    assert np.all((imgs == 0.0).sum(axis=(1, 2)) == 24)
    rows, cols = np.nonzero(imgs == 1.0)[1:]
    assert np.all(np.abs(rows - 2) <= 1) and np.all(np.abs(cols - 2) <= 1)
    # Con 300 muestras aparecen los 9 desplazamientos posibles.
    assert len(set(zip(rows, cols, strict=True))) == 9


def test_shift_rellena_con_el_minimo_de_la_fila():
    # Imagen escalada a [−1, 1]: el fondo es −1, no 0.
    shape = (4, 4)
    X = one_pixel_images(100, shape, (0, 0), bg=-1.0)
    out = RandomShift(max_px=1, image_shape=shape)(X, rng())
    assert set(np.unique(out)) <= {-1.0, 1.0}
    # El píxel de la esquina puede salirse de la imagen, pero nunca aparece más de uno.
    assert np.all((out == 1.0).sum(axis=1) <= 1)
    assert np.any((out == 1.0).sum(axis=1) == 0)


def test_shift_cero_es_identidad():
    X = rng(1).uniform(size=(5, 12))
    np.testing.assert_array_equal(RandomShift(max_px=0, image_shape=(3, 4))(X, rng()), X)


def test_shift_forma_incompatible_falla():
    with pytest.raises(ValueError):
        RandomShift(max_px=1, image_shape=(3, 3))(np.zeros((2, 8)), rng())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_px": -1, "image_shape": (3, 3)},
        {"max_px": 3, "image_shape": (3, 3)},
        {"max_px": 1, "image_shape": (3,)},
        {"max_px": 1, "image_shape": (0, 3)},
    ],
)
def test_shift_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        RandomShift(**kwargs)


# --- Compose y registro ---


def test_compose_aplica_en_orden_con_el_mismo_generador():
    X = rng(1).uniform(size=(10, 9))
    shift, noise = RandomShift(max_px=1, image_shape=(3, 3)), GaussianNoise(sigma=0.1)
    g = rng(5)
    expected = noise(shift(X, g), g)
    np.testing.assert_array_equal(Compose([shift, noise])(X, rng(5)), expected)


def test_get_augmentation_desde_config():
    aug = get_augmentation({"kind": "gaussian_noise", "sigma": 0.2, "clip": [0, 1]})
    assert isinstance(aug, GaussianNoise) and aug.sigma == 0.2

    aug = get_augmentation({"kind": "random_shift", "max_px": 1, "image_shape": [8, 8]})
    assert isinstance(aug, RandomShift) and aug.image_shape == (8, 8)

    aug = get_augmentation(
        [
            {"kind": "random_shift", "max_px": 1, "image_shape": [8, 8]},
            {"kind": "gaussian_noise", "sigma": 0.05},
        ]
    )
    assert isinstance(aug, Compose)
    assert [type(a) for a in aug.augmentations] == [RandomShift, GaussianNoise]
    assert set(AUGMENTATIONS) == {"gaussian_noise", "random_shift"}


@pytest.mark.parametrize(
    "config",
    [
        {"kind": "rotation", "degrees": 10},
        {"sigma": 0.1},
        {"kind": "gaussian_noise", "sigma": 0.1, "mean": 0.0},
    ],
    ids=["kind_desconocido", "sin_kind", "parametro_desconocido"],
)
def test_get_augmentation_config_invalida_falla(config):
    with pytest.raises((ValueError, TypeError)):
        get_augmentation(config)
