import jax
from jax import numpy as jnp
import flax
from flax.training import train_state
import flax.nnx
import optax
import chex

from mlx import data as dx

import numpy as np
import random
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import mlflow

from Generator import ConditionalGenerator, Discriminator


class TrainState(train_state.TrainState):
    batch_stats: dict


def parse_arguments() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser(description='Parse input arguments')

    parser.add_argument('--experiment-name', type=str, default='Probabilistic semi-supervised')
    parser.add_argument('--run-description', type=str, default=None)

    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nz', type=int, default=100)

    parser.add_argument(
        '--train-groundtruth-file',
        type=str,
        help='Path to ground truth of training set'
    )

    parser.add_argument('--num-classes', type=int, help='Number of classes')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='The total number of epochs to run'
    )

    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)

    parser.add_argument('--run-id', type=str, default=None, help='Run ID in MLFlow')

    parser.add_argument(
        '--jax-platform',
        type=str,
        default='cpu',
        help='cpu, cuda or tpu'
    )
    parser.add_argument(
        '--mem-frac',
        type=float,
        default=0.9,
        help='Percentage of GPU memory allocated for Jax'
    )

    parser.add_argument('--prefetch-size', type=int, default=8)
    parser.add_argument('--num-threads', type=int, default=4)

    parser.add_argument('--tqdm', dest='tqdm', action='store_true')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
    parser.set_defaults(tqdm=True)

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument(
        '--tracking-uri',
        type=str,
        default='http://127.0.0.1:8080',
        help='MLFlow server'
    )

    return parser.parse_args()


def get_dataset(dataset_file: Path | str) -> dx._c.Buffer:
    # load json data
    with open(file=dataset_file, mode='r') as f:
        # load a list of dictionaries
        json_data = json.load(fp=f)

    # list of dictionaries, each dictionary is a sample
    data_dicts = [
        dict(file=data_dict['file'].encode('ascii'), label=data_dict['label']) \
            for data_dict in json_data
    ]

    # load image dataset without batching nor shuffling
    dset = (
        dx.buffer_from_vector(data=data_dicts)
        .load_image(key='file', output_key='image')
        .image_resize(key='image', w=64, h=64)
        .key_transform(key='image', func=lambda x: x.astype('float32') / 255)
        .key_transform(key='label', func=lambda x: x.astype('int32'))
    )

    return dset


def preparare_dataset(dataset: dx._c.Buffer, shuffle: bool) -> dx._c.Stream:
    """
    """
    if shuffle:
        dset = dataset.shuffle()
    else:
        dset = dataset

    dset = (
        dset
        # .to_stream()
        .batch(batch_size=args.batch_size)
        .prefetch(prefetch_size=args.prefetch_size, num_threads=args.num_threads)
    )

    return dset


def train_discriminator(
    x: chex.Array,
    y: chex.Array | list[int],
    state_g: TrainState,
    state_d: TrainState,
    key: jax.random.PRNGKey
) -> tuple[TrainState, chex.Scalar]:
    """
    """
    def loss_discriminator(
        params: flax.core.frozen_dict.FrozenDict
    ) -> tuple[chex.Scalar, flax.core.frozen_dict.FrozenDict]:
        """Loss of discriminator"""
        z = jax.random.normal(key=key, shape=(len(y), 1, 1, args.nz))
        x_fake, _ = state_g.apply_fn(
            variables={'params': state_g.params, 'batch_stats': state_g.batch_stats},
            x=z,
            y=y,
            train=True,
            mutable=['batch_stats']
        )
        fake_labels = jnp.zeros(shape=(len(y),))
        fake_logits, batch_stats_new = state_d.apply_fn(
            variables={'params': params, 'batch_stats': state_d.batch_stats},
            x=x_fake,
            train=True,
            mutable=['batch_stats']
        )
        fake_loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=fake_logits,
            labels=fake_labels
        )

        real_labels = jnp.ones(shape=(len(x),))
        real_logits, batch_stats_new = state_d.apply_fn(
            variables={'params': params, 'batch_stats': batch_stats_new['batch_stats']},
            x=x,
            train=True,
            mutable=['batch_stats']
        )
        real_loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=real_logits,
            labels=real_labels
        )
        real_loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=real_logits,
            labels=real_labels
        )
        loss = jnp.mean(a=fake_loss + real_loss, axis=0)

        return loss, batch_stats_new

    grad_value_fn_d = jax.value_and_grad(
        fun=loss_discriminator,
        argnums=0,
        has_aux=True
    )
    (loss_d, batch_stats_d), grads_d = grad_value_fn_d(state_d.params)

    state_d = state_d.apply_gradients(grads=grads_d)
    state_d = state_d.replace(batch_stats=batch_stats_d['batch_stats'])

    return state_d, loss_d


def train_generator(
    y: chex.Array | list[int],
    state_g: TrainState,
    state_d: TrainState,
    key: jax.random.PRNGKey
) -> tuple[TrainState, chex.Scalar]:
    """
    """
    def loss_generator(
        params: flax.core.frozen_dict.FrozenDict,
        y: chex.Array | list[int],
    ) -> tuple[chex.Scalar, flax.core.frozen_dict.FrozenDict]:
        """Loss of generator"""
        z = jax.random.normal(key=key, shape=(len(y), 1, 1, args.nz))
        x_fake, batch_stats_new = state_g.apply_fn(
            variables={'params': params, 'batch_stats': state_g.batch_stats},
            x=z,
            y=y,
            train=True,
            mutable=['batch_stats']
        )

        logits, _ = state_d.apply_fn(
            variables={'params': state_d.params, 'batch_stats': state_d.batch_stats},
            x=x_fake,
            train=True,
            mutable=['batch_stats']
        )
        y = jnp.ones(shape=(len(x_fake),))
        loss = optax.losses.sigmoid_binary_cross_entropy(
            logits=logits,
            labels=y
        )
        loss = jnp.mean(a=loss, axis=0)

        return loss, batch_stats_new

    grad_value_fn_g = jax.value_and_grad(
        fun=loss_generator,
        argnums=0,
        has_aux=True
    )
    (loss_g, batch_stats_g), grads_g = grad_value_fn_g(state_g.params, y)

    state_g = state_g.apply_gradients(grads=grads_g)
    state_g = state_g.replace(batch_stats=batch_stats_g['batch_stats'])

    return state_g, loss_g


@jax.jit
def train_step(
    x: chex.Array,
    y: chex.Array | list[int],
    state_g: TrainState,
    state_d: TrainState,
    key_g: jax.random.PRNGKey,
    key_d: jax.random.PRNGKey
) -> tuple[TrainState, chex.Scalar, TrainState, chex.Scalar]:

    state_d, loss_d = train_discriminator(
        x=x, y=y, state_g=state_g, state_d=state_d, key=key_d
    )

    state_g, loss_g = train_generator(y=y, state_g=state_g, state_d=state_d, key=key_g)

    return state_g, loss_g, state_d, loss_d


def train(
    dataset: dx._c.Buffer,
    state_g: TrainState,
    state_d: TrainState
) -> tuple[TrainState, chex.Scalar, TrainState, chex.Scalar]:
    """
    """
    dset = (
        dataset
        .shuffle()
        .to_stream()
        .batch(batch_size=args.batch_size)
        .prefetch(prefetch_size=args.prefetch_size, num_threads=args.num_threads)
    )

    loss_g_accum = flax.nnx.metrics.Average()
    loss_d_accum = flax.nnx.metrics.Average()

    for samples in tqdm(
        iterable=dset,
        desc='train',
        total=len(dataset)//args.batch_size + 1,
        position=2,
        leave=False,
        disable=not args.tqdm
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)
        y = jnp.asarray(a=samples['label'], dtype=jnp.int32)

        args.key, key_g, key_d = jax.random.split(key=args.key, num=3)

        state_g, loss_g, state_d, loss_d = train_step(
            x,
            y,
            state_g,
            state_d,
            key_g,
            key_d
        )

        for loss_val, loss_accum in zip((loss_g, loss_d), (loss_g_accum, loss_d_accum)):
            loss_accum.update(values=loss_val)

    return state_g, loss_g_accum.compute(), state_d, loss_d_accum.compute()


def initialise_generator(
    ngf: int,
    nz: int,
    num_classes: int,
    key: jax.random.PRNGKey
) -> TrainState:
    key1, key2 = jax.random.split(key=key, num=2)

    generator = ConditionalGenerator(ngf=ngf, nc=3, num_classes=num_classes)
    params = generator.init(
        rngs=key1,
        x=jax.random.normal(key=key2, shape=(num_classes, 1, 1, nz)),
        # y=jax.random.randint(key=key, shape=(2,), minval=0, maxval=num_classes),
        y=jnp.arange(num_classes),
        train=False
    )
    tx = optax.adam(learning_rate=args.lr, b1=0.5)
    state = TrainState.create(
        apply_fn=generator.apply,
        params=params['params'],
        batch_stats=params['batch_stats'],
        tx=tx
    )

    return state


def initialise_discriminator(
    ndf: int,
    image_shape: tuple[int, int, int],
    key: jax.random.PRNGKey
) -> TrainState:
    key1, key2 = jax.random.split(key=key, num=2)

    discriminator = Discriminator(ndf=ndf)
    params = discriminator.init(
        rngs=key1,
        x=jax.random.uniform(key=key2, shape=(1,) + image_shape),
        train=False
    )
    tx = optax.adam(learning_rate=args.lr, b1=0.5)
    state = TrainState.create(
        apply_fn=discriminator.apply,
        params=params['params'],
        batch_stats=params['batch_stats'],
        tx=tx
    )

    return state


def main() -> None:
    args.key = jax.random.key(seed=random.randint(a=0, b=1_000))

    # region DATASET
    dset_train = get_dataset(dataset_file=args.train_groundtruth_file)
    # endregion

    # region MODELS
    args.key, key_g, key_d = jax.random.split(key=args.key, num=3)
    state_d = initialise_discriminator(
        ndf=args.ndf,
        image_shape=dset_train[0]['image'].shape,
        key=key_d
    )
    state_g = initialise_generator(
        ngf=args.ngf,
        nz=args.nz,
        num_classes=args.num_classes,
        key=key_g
    )
    # endregion

    n_z = 5
    z = jax.random.normal(key=jax.random.key(seed=44), shape=(n_z, 1, 1, args.nz))
    y_z = jax.random.randint(
        key=jax.random.key(seed=44),
        shape=(n_z,),
        minval=0,
        maxval=args.num_classes
    )

    # enable MLFlow tracking
    mlflow.set_tracking_uri(uri=args.tracking_uri)
    mlflow.set_experiment(experiment_name=args.experiment_name)
    mlflow.set_system_metrics_sampling_interval(interval=60)
    mlflow.set_system_metrics_samples_before_logging(samples=1)
    with mlflow.start_run(
        log_system_metrics=True,
        description=args.run_description
    ) as mlflow_run:
        # log hyper-parameters
        mlflow.log_params(
            params={key: args.__dict__[key] \
                for key in args.__dict__ \
                    if isinstance(args.__dict__[key], (int, bool, str, float))}
        )

        for epoch_id in tqdm(
            iterable=range(args.num_epochs),
            desc='total',
            leave=True,
            position=1,
            disable=not args.tqdm
        ):
            state_g, loss_g_accum, state_d, loss_d_accum = train(
                dataset=dset_train,
                state_g=state_g,
                state_d=state_d
            )

            mlflow.log_metrics(
                metrics={
                    'loss_g': loss_g_accum,
                    'loss_d': loss_d_accum
                },
                step=epoch_id + 1
            )

            # generate some images
            x_fake, _ = state_g.apply_fn(
                variables={
                    'params': state_g.params,
                    'batch_stats': state_g.batch_stats
                },
                x=z,
                y=y_z,
                train=True,
                mutable=['batch_stats']
            )
            for i in range(len(x_fake)):
                mlflow.log_image(
                    image=np.asarray(a=x_fake[i], dtype=np.float32),
                    key='image_{:d}'.format(i),
                    step=epoch_id + 1
                )

    return None


if __name__ == '__main__':
    # parse input arguments
    args = parse_arguments()

    # set jax memory allocation
    jax.config.update(name='jax_platforms', val=args.jax_platform)
    assert args.mem_frac < 1. and args.mem_frac > 0.
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(args.mem_frac)

    # disable mlflow's tqdm
    os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'

    main()
