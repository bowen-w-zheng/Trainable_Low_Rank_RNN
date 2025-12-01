"""Test if empty trainable_params dict prevents training."""
import jax
import jax.numpy as jnp
import optax

# Simulate what happens when all training flags are False
def test_empty_trainable_params():
    print("=" * 60)
    print("Testing empty trainable_params behavior")
    print("=" * 60)

    # Create some fixed parameters
    fixed_params = {
        'M': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'N': jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        'w': jnp.array([1.0, 2.0])
    }

    # Empty trainable params (all flags False)
    trainable_params = {}

    print(f"\nInitial fixed_params keys: {fixed_params.keys()}")
    print(f"Initial trainable_params keys: {trainable_params.keys()}")
    print(f"Number of trainable params: {len(trainable_params)}")

    # Define a simple loss function
    def loss_fn(trainable, fixed):
        # Use fixed params to compute loss
        M = trainable.get('M', fixed.get('M'))
        N = trainable.get('N', fixed.get('N'))
        w = trainable.get('w', fixed.get('w'))

        # Simple loss: sum of all elements
        loss = jnp.sum(M) + jnp.sum(N) + jnp.sum(w)
        return loss

    # Create optimizer
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(trainable_params)

    print(f"\nOptimizer state initialized: {type(opt_state)}")

    # Compute initial loss
    init_loss = loss_fn(trainable_params, fixed_params)
    print(f"Initial loss: {init_loss}")

    # Simulate training steps
    for step in range(5):
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(trainable_params, fixed_params)

        print(f"\n--- Step {step} ---")
        print(f"Loss: {loss}")
        print(f"Grads keys: {grads.keys()}")
        print(f"Grads: {grads}")

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)

        print(f"Updates keys: {updates.keys()}")
        print(f"Updates: {updates}")
        print(f"Trainable params after update keys: {trainable_params.keys()}")
        print(f"Trainable params after update: {trainable_params}")

    # Final loss
    final_loss = loss_fn(trainable_params, fixed_params)
    print(f"\n{'=' * 60}")
    print(f"Final loss: {final_loss}")
    print(f"Loss changed: {init_loss != final_loss}")
    print(f"Expected: Loss should NOT change (still {init_loss})")
    print(f"{'=' * 60}")

    if init_loss == final_loss:
        print("\n✅ CORRECT: Empty trainable_params prevents training")
    else:
        print("\n❌ BUG: Empty trainable_params still allows training!")

if __name__ == "__main__":
    test_empty_trainable_params()
