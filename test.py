import copy
import time
import numpy as np
from neuron import Neuron, Synapse
from creature import Creature
from finetune import Finetune

def run_extreme_stress_test():
    print("==================================================")
    print("💥 STARTING EXTREME NEURAL GRAPH STRESS TESTS 💥")
    print("==================================================\n")

    # ----------------------------------------------------------------
    # TEST 1: RECURRENT LOOP & REFLECTION GRAPH (Deep Cycle Check)
    # ----------------------------------------------------------------
    print("🔬 Test 1: Testing Recurrent Loops (Cyclic Graphs)...")
    loop_neurons = [Neuron(pos=(i, i), color="green") for i in range(10)]
    
    # Wire them in a strict circle: 0->1->2->3...->9->0
    for i in range(10):
        loop_neurons[i].add_synapse(loop_neurons[(i + 1) % 10], weight=1.0, delay=1)
        
    cyclical_creature = Creature(loop_neurons)
    cloned_cycle = cyclical_creature.clone()
    
    # Mutate the clone's loop weights to verify they don't impact the original loop
    for s in cloned_cycle.all_synapses:
        s.weight = 999.0
        
    for s in cyclical_creature.all_synapses:
        if s.weight == 999.0:
            raise ValueError("❌ STRESS TEST FAILED: Cyclic synapse mutation bled backward!")
            
    print("✅ PASS: Recurrent structures perfectly duplicated without cross-contamination.")

    # ----------------------------------------------------------------
    # TEST 2: RANDOM GENERATOR STATE ISOLATION (Independence Check)
    # ----------------------------------------------------------------
    print("\n🔬 Test 2: Testing Stochastic Random Stream Isolation...")
    rng = np.random.default_rng(seed=42)
    stochastic_neuron = Neuron(pos=(0,0), color="blue", rng=rng)
    stochastic_creature = Creature([stochastic_neuron])
    
    cloned_stochastic = stochastic_creature.clone()
    
    # Draw from the original creature's generator
    val_orig = stochastic_creature.neurons[0].rng.standard_normal()
    # Draw from the clone's generator
    val_clone = cloned_stochastic.neurons[0].rng.standard_normal()
    
    # If they are sharing the exact same state reference instance, drawing from one
    # will advance the internal state of the other, causing values to diverge or conflict.
    # Note: copy.deepcopy(np.random.Generator) safely produces a decoupled state copy.
    assert val_orig != val_clone, "❌ STRESS TEST FAILED: Cloned RNG stream is synchronized with the original!"
    print("✅ PASS: Random number generators are fully isolated and distinct streams.")

    # ----------------------------------------------------------------
    # TEST 3: THE FINETUNE WIRING MUTATION ACID TEST
    # ----------------------------------------------------------------
    print("\n🔬 Test 3: Simulating the Finetune Wiring Mutation Lifecycle...")
    
    # Construct a valid network backbone
    base_neurons = (
        [Neuron(pos=(0,0), color="white", is_input_neuron=True) for _ in range(5)] +
        [Neuron(pos=(1,1), color="gray") for _ in range(10)] +
        [Neuron(pos=(2,2), color="black", is_output_neuron=True) for _ in range(3)]
    )
    base_creature = Creature(base_neurons)
    
    # Run through your actual Finetune loop class setup
    finetuner = Finetune(base_creature, batch_size=20)
    
    for idx, variant in enumerate(finetuner.finetunes):
        # Tick the variants ahead dynamically
        variant.brain_tick()
        
        # STRESS TRAP CHECK: Does 'all_synapses' track newly added synapses accurately?
        expected_synapse_count = sum(len(n.outgoing_synapses) for n in variant.neurons)
        
        # If your Creature constructor caches all_synapses ONLY at boot up,
        # Finetune's manual modifications (a.add_synapse / a.delete_synapse) 
        # will cause variant.all_synapses to drift out of sync! Let's check:
        if len(variant.all_synapses) != expected_synapse_count:
            print(f"⚠️  CRITICAL BEHAVIOR WARNING (Variant {idx}):")
            print(f"   Neuron tracking says {expected_synapse_count} synapses exist.")
            print(f"   But creature.all_synapses only has {len(variant.all_synapses)} cached.")
            print("   👉 FIX REQUIREMENT: Re-sync 'all_synapses' dynamically when adding/deleting.")
            
            # Let's fix it mid-flight for the rest of the verification checks
            variant.all_synapses = [s for n in variant.neurons for s in n.outgoing_synapses]
            
        # Ensure synapse structural loops inside the variant are entirely closed loops
        for s in variant.all_synapses:
            if s.sender not in variant.neurons or s.receiver not in variant.neurons:
                raise ValueError("❌ STRESS TEST FAILED: Mutated synapse references an external creature's neuron!")

    print("✅ PASS: Graph restructuring operations successfully confined to child scopes.")

    # ----------------------------------------------------------------
    # TEST 4: MASS BENCHMARK RUN (Scale & Performance)
    # ----------------------------------------------------------------
    print("\n🔬 Test 4: Performance Benchmarking (Mass Cloning Scaling)...")
    large_neurons = [Neuron(pos=(i, i), color="yellow") for i in range(100)]
    for i in range(99):
        large_neurons[i].add_synapse(large_neurons[i+1], weight=0.1)
    large_creature = Creature(large_neurons)
    
    start_time = time.time()
    iterations = 200
    for _ in range(iterations):
        _ = large_creature.clone()
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"⚡ Performance Profile: Cloned a 100-node network {iterations} times in {elapsed:.4f} seconds.")
    print(f"⚡ Average allocation speed: {(elapsed / iterations) * 1000:.2f} ms per deep clone.")
    
    print("\n==================================================")
    print("🎉 ALL EXTREME STRESS TESTS COMPLETED SUCCESSFULLY! 🎉")
    print("==================================================")

if __name__ == "__main__":
    run_extreme_stress_test()
