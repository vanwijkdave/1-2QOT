
# Import the two programs to form the application ran
import time

from application import AliceProgram, BobProgram

from squidasm.run.stack.config import (StackNetworkConfig, DepolariseLinkConfig, LinkConfig)
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager

from error_correction import UniReedSolomonCorrection, LDPCErrorCorrection
from hashing import CarterWegmanHash, ToeplitzHash

from multiprocessing import Pool

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# import network configuration from file
cfg = StackNetworkConfig.from_file("./params/qdevice_params_optimistic.yaml")
depolarise_config = DepolariseLinkConfig.from_file("./params/depolarise_link_config.yaml")
link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
# Replace link from YAML file with new depolarise link
cfg.links = [link]


# Set logging
LogManager.set_log_level("ERROR")
# Disable logging to terminal
logger = LogManager.get_stack_logger()
logger.handlers = []
# Enable logging to file
LogManager.log_to_file("info.log")

# Security Parameters for OT
multiplier = 3
lam_OT  = 256 * multiplier     # length of bitstring
lam_HS  = 20 * multiplier     # Hash size
lam_BS  = 20 * multiplier      # Block size for error correction
lam_PQS = 256     # Size of nonce added
Q_tol   = 0.025     # Tolerance for fault in bit commitment
chosen_message = 1     # Bob's choice of message, either 0 or 1
error_correction = LDPCErrorCorrection()
hash_function = CarterWegmanHash()
messages = ["hi, i'm alice, encoding a message", "you might see me, or the other message, i don't know which one"]


def calculate_qber_for_fidelity():
    link_fidelity_list = np.arange(0.3, 1.0, step=0.05)
    n = 20
    qber_result_list = []
    qber_error_bars = []
    messages = ["hi, i'm alice, encoding a message", "you might see me, or the other message, i don't know which one"]
    for fidelity in link_fidelity_list:
        depolarise_config.fidelity = fidelity
        # Create instances of programs to run
        alice_program = AliceProgram(messages=messages, lam_OT=lam_OT, lam_HS=lam_HS, Q_tol=Q_tol, lam_PQS=lam_PQS, error_correction=error_correction, hash_function=hash_function)
        bob_program = BobProgram(lam_OT=lam_OT, lam_HS=lam_HS, lam_PQS=lam_PQS, chosen_message=chosen_message, error_correction=error_correction, hash_function=hash_function)

        res_alice, res_bob = run(config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, 
            num_times=n)
        alice_qber = [res_alice[i]["qber"] for i in range(n)]
        qber_result_list.append(sum(alice_qber) / len(alice_qber))
        # error bars
        qber_error_bars.append(np.std(alice_qber) / np.sqrt(len(alice_qber)))
    # Run the simulation. Programs argument is a mapping of network node labels to programs to run on that node
    # measure time

    # plot for the qber avg
    xpoints = link_fidelity_list
    ypoints = qber_result_list
    plt.errorbar(xpoints, ypoints, yerr=qber_error_bars, marker='o')
    plt.title("Average QBER vs fidelity")
    plt.xlabel("Fidelity")
    plt.ylabel("Average QBER (%)")
    plt.xticks(xpoints)
    plt.grid()
    plt.draw()
    # save as pdf
    datetime = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"qber_vs_fidelity_{datetime}.pdf")
    # save raw data for later analysis / LaTeX plots
    np.savez(f"qber_vs_fidelity_data_{datetime}.npz", xpoints=xpoints, ypoints=ypoints, qber_error_bars=qber_error_bars)
    plt.show()


def run_simulation_for_leakage_rate(args):
    fidelity, n = args
    fidelity = float(np.clip(fidelity, 0.0, 1.0))  # guard against float drift

    depolarise_config = DepolariseLinkConfig.from_file("./params/depolarise_link_config.yaml")
    cfg = StackNetworkConfig.from_file("./params/qdevice_params_optimistic.yaml")
    link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
    depolarise_config.fidelity = fidelity
    cfg.links = [link]


    messages = ["hi, i'm alice, encoding a message", "you might see me, or the other message, i don't know which one"]
    max_iter = 200
    # leakage_rate = round(0.8 - 0.84 * fidelity, 2)  # Start with a starting point based on heuristics
    if fidelity < 0.85:
        leakage_rate = 0.5
    elif fidelity < 0.9:
        leakage_rate = 0.45
    elif fidelity < 0.95:
        leakage_rate = 0.25
    elif fidelity < 0.97:
        leakage_rate = 0.15
    else:
        leakage_rate = 0.01

    succes_rate = 0
    iterations = 0
    leakage_rates = {}
    while succes_rate < 1 and iterations < max_iter:
        leakage_rate += 0.01
        alice_program = AliceProgram(messages=messages, lam_OT=lam_OT, lam_HS=lam_HS, Q_tol=Q_tol, lam_PQS=lam_PQS, error_correction=error_correction, hash_function=hash_function, leakage_rate=leakage_rate)
        bob_program = BobProgram(lam_OT=lam_OT, lam_HS=lam_HS, lam_PQS=lam_PQS, chosen_message=chosen_message, error_correction=error_correction, hash_function=hash_function, leakage_rate=leakage_rate)

        res_alice, res_bob = run(config=cfg, programs={"Alice": alice_program, "Bob": bob_program},
            num_times=n)
        succes_index = [i for i in range(n) if res_alice[i]['m0'] == res_bob[i]['m0'] or res_alice[i]['m1'] == res_bob[i]['m1']]
        qber = [res_alice[i]['qber'] for i in range(n)]
        succes_rate = len(succes_index) / n
        print(f"Fidelity: {fidelity}, Leakage Rate: {leakage_rate}, Success Rate: {succes_rate}")

        # if succes_rate < 0.7:
        #     leakage_rate = round(leakage_rate + 0.01, 5)

        if succes_rate < 0.5:
            leakage_rate = round(leakage_rate + 0.03, 5)

        if succes_rate < 0.3:
            leakage_rate = round(leakage_rate + 0.05, 5)

        if succes_rate < 0.1:
            leakage_rate = round(leakage_rate + 0.07, 5)

        if succes_rate > 0.8 and 0.8 not in leakage_rates:
            leakage_rates[0.8] = leakage_rate

        if succes_rate > 0.90 and 0.9 not in leakage_rates:
            leakage_rates[0.9] = leakage_rate

        if succes_rate == 1:
            leakage_rates[1.0] = leakage_rate

        iterations += 1

    return leakage_rates


def calculate_minimum_leakage_rate():
    link_fidelity_list = np.linspace(0.8, 1, num=12)
    n = 20
    leakage_rates = []
    leakage_rates_for_success = {0.8: [], 0.9: [], 1.0: []}
    # for fidelity in link_fidelity_list:
    #     leakage_rates.append(run_simulation_for_leakage_rate(fidelity, n))
    # now the above two lines multithreaded

    args = [(f, n) for f in link_fidelity_list]
    with Pool() as pool:
        leakage_rates = pool.map(run_simulation_for_leakage_rate, args)

    # currently the leakage rates is a list of dicts per fidelity containing the values for different p_succes [{0.9: 1.1}, {0.95: 1.4}, ...]
    # we want to map per leakage rate the corresponding fidelity, 
    for i, fidelity in enumerate(link_fidelity_list):
        values = leakage_rates[i]
        for key in values:
            leakage_rates_for_success[key].append((fidelity, values[key]))

    # formula for fidelity to QBER= 66-66*fidelity, based on heuristics and experimental data from the simulation, not exact but gives a good starting point for the search of the minimum leakage rate
    qber_list = [round(66 - 66 * f, 1) for f in link_fidelity_list]
    # plot for the leakage rate
    xpoints = link_fidelity_list
    # ypoints = leakage_rates
    # multiple lines for different p_succes
    plt.plot(xpoints, [f[1] for f in leakage_rates_for_success[0.8]], marker='o', label='p_success = 0.8')
    plt.plot(xpoints, [f[1] for f in leakage_rates_for_success[0.9]], marker='o', label='p_success = 0.9')
    plt.plot(xpoints, [f[1] for f in leakage_rates_for_success[1.0]], marker='o', label='p_success = 1.0')
    plt.title("Minimum Leakage Rate for different fidelity")
    plt.xlabel("Fidelity")
    plt.ylabel("Leakage Rate (L/n)")
    plt.grid()
    plt.legend()

    plt.draw()
    # save as pdf
    datetime = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"leakage_rate_vs_fidelity_{datetime}.pdf")
    # save for latex
    plt.show()
    # save data

    plt.plot(xpoints, [f[1] for f in leakage_rates_for_success[0.8]], marker='o', label='p_success = 0.8')
    plt.plot(xpoints, [f[1] for f in leakage_rates_for_success[0.9]], marker='o', label='p_success = 0.9')
    plt.plot(xpoints, [f[1] for f in leakage_rates_for_success[1.0]], marker='o', label='p_success = 1.0')
    plt.title("Minimum Leakage Rate for different QBER")
    plt.xlabel("QBER (%)")
    plt.ylabel("Leakage Rate (L/n)")
    plt.grid()
    plt.legend()
    plt.draw()
    # save as pdf
    datetime = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"leakage_rate_vs_QBER_{datetime}.pdf")
    # save for latex
    plt.show()
    np.savez(f"leakage_rate_vs_fidelity_data_{datetime}.npz", qber_list=qber_list, leakage_rates_for_success=leakage_rates_for_success)


import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Worker function: does ONE (scenario, leakage_rate) run
def _run_single_leakage(scenario, leakage_rate, n):
    cfg = StackNetworkConfig.from_file(f"./params/qdevice_params_{scenario}.yaml")
    depolarise_config = DepolariseLinkConfig.from_file(f"./params/link_params_{scenario}.yaml")
    link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
    # Replace link from YAML file with new depolarise link
    cfg.links = [link]

    alice_program = AliceProgram(
        messages=messages,
        lam_OT=lam_OT,
        lam_HS=lam_HS,
        Q_tol=Q_tol,
        lam_PQS=lam_PQS,
        error_correction=error_correction,
        hash_function=hash_function,
        leakage_rate=leakage_rate,
    )
    bob_program = BobProgram(
        lam_OT=lam_OT,
        lam_HS=lam_HS,
        lam_PQS=lam_PQS,
        chosen_message=chosen_message,
        error_correction=error_correction,
        hash_function=hash_function,
        leakage_rate=leakage_rate,
    )

    res_alice, res_bob = run(
        config=cfg,
        programs={"Alice": alice_program, "Bob": bob_program},
        num_times=n,
    )

    succes_index = [
        i for i in range(n)
        if res_alice[i]['m0'] == res_bob[i]['m0']
        or res_alice[i]['m1'] == res_bob[i]['m1']
    ]
    qber = [res_alice[i]['qber'] for i in range(n)]
    succes_rate = len(succes_index) / n

    # Return everything needed to reconstruct results in the main process
    print(f"Finished scenario {scenario} with leakage rate {leakage_rate}: success rate {succes_rate}, QBER {qber}")
    return scenario, leakage_rate, succes_rate, qber


def calculate_leakage_succes_rate():
    # x_leakage_rate_list = np.arange(0.1, 1.0, step=0.1)
    # we want the x axis like above, but focus 3/4 of the points between 0.1 and 0.5, and 1/4 of the points between 0.5 and 1.0, so we can see the difference in succes rate more clearly in the lower leakage rate region
    x_leakage_rate_list = np.concatenate([
        np.arange(0.1, 0.5, step=0.03),
        np.arange(0.5, 1.0, step=0.1)
    ])
    y_succes_rate_list = {}
    n = 15
    scenarios = ["current", "optimistic"]

    for scenario in scenarios:
        y_succes_rate_list[scenario] = []

        # Launch parallel jobs for all leakage rates in this scenario
        results = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_run_single_leakage, scenario, leakage_rate, n): leakage_rate
                for leakage_rate in x_leakage_rate_list
            }

            for future in as_completed(futures):
                scen, leakage_rate, succes_rate, qber = future.result()
                results.append((leakage_rate, succes_rate, qber))

        # Sort results to match x_leakage_rate_list order
        results.sort(key=lambda t: t[0])

        # Store success rates and print logs
        for leakage_rate, succes_rate, qber in results:
            y_succes_rate_list[scenario].append(succes_rate)
            print(f"Scenario: {scenario}, Leakage Rate: {leakage_rate}, "
                  f"Success Rate: {succes_rate}, QBER: {qber}")

    # Plot the results
    for scenario in scenarios:
        plt.plot(x_leakage_rate_list, y_succes_rate_list[scenario], marker='o', label=f"{scenario.capitalize()}")
    plt.title("Success Rate vs Leakage Rate - LDPC Error Correction")
    plt.xlabel("Leakage Rate (L/t)")
    plt.ylabel("Success Rate (%)")
    plt.grid()
    plt.legend()
    plt.draw()
    # save as pdf
    datetime = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"success_rate_vs_leakage_rate_{datetime}.pdf")
    plt.show()
    # save data
    np.savez(f"success_rate_vs_leakage_rate_data_{datetime}.npz", x_leakage_rate_list=x_leakage_rate_list, y_succes_rate_list=y_succes_rate_list)

from concurrent.futures import ProcessPoolExecutor, as_completed

def _run_single_time_taken(scenario, lam_ot, n):
    """
    Run the protocol n times for a given scenario and message length (lam_ot),
    and return average total time per run, split into quantum and post-processing parts.
    """
    # Load scenario-specific configs
    cfg = StackNetworkConfig.from_file(f"./params/qdevice_params_{scenario}.yaml")
    depolarise_config = DepolariseLinkConfig.from_file(f"./params/link_params_{scenario}.yaml")
    link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
    cfg.links = [link]

    # Use LDPC + Carter-Wegman as specified
    error_correction = LDPCErrorCorrection()
    hash_function = CarterWegmanHash()

    alice_program = AliceProgram(
        lam_OT=lam_ot,
        lam_HS=lam_HS,
        Q_tol=Q_tol,
        lam_PQS=lam_PQS,
        error_correction=error_correction,
        hash_function=hash_function,
        # you can add leakage_rate here if needed, but for timing it's not required
    )
    bob_program = BobProgram(
        lam_OT=lam_ot,
        lam_HS=lam_HS,
        lam_PQS=lam_PQS,
        chosen_message=chosen_message,
        error_correction=error_correction,
        hash_function=hash_function,
    )

    # Run with statistics enabled
    stats, res = run(
        config=cfg,
        programs={"Alice": alice_program, "Bob": bob_program},
        num_times=n,
        measure_stats=True,
    )
    res_alice, res_bob = res

    # elapsed_sim_time is in ns (comment in your code), convert to seconds
    sim_time_total_s = stats.data["elapsed_sim_time"] * 1e-9
    quantum_time_per_run = sim_time_total_s / n

    # Bob's post-processing time is per run and already in seconds
    post_times = [res_bob[i]["post_process_time"] for i in range(n)]
    post_time_avg = float(np.mean(post_times))

    total_time_avg = quantum_time_per_run + post_time_avg

    return scenario, lam_ot, total_time_avg, quantum_time_per_run, post_time_avg


def calculate_time_taken():
    # Message lengths to test (in bits). Adjust this list as needed.
    lam_ot_list = np.array([128, 256, 512, 1024, 2048, 4096])

    # Number of protocol executions per data point (per scenario, per lam_OT)
    n = 5

    scenarios = ["current", "optimistic"]

    # Store results: average times per run
    total_times = {scenario: [] for scenario in scenarios}
    quantum_times = {scenario: [] for scenario in scenarios}
    post_times = {scenario: [] for scenario in scenarios}

    for scenario in scenarios:
        results = []

        # Parallelise over message lengths for this scenario
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_run_single_time_taken, scenario, lam_ot, n): lam_ot
                for lam_ot in lam_ot_list
            }

            for future in as_completed(futures):
                scen, lam_ot, total_time, quantum_time_per_run, post_time_avg = future.result()
                results.append((lam_ot, total_time, quantum_time_per_run, post_time_avg))

        # Sort results by message length to align with lam_ot_list
        results.sort(key=lambda t: t[0])

        for lam_ot, total_time, quantum_time_per_run, post_time_avg in results:
            total_times[scenario].append(total_time)
            quantum_times[scenario].append(quantum_time_per_run)
            post_times[scenario].append(post_time_avg)
            print(
                f"Scenario: {scenario}, lam_OT: {lam_ot}, "
                f"total_time: {total_time:.6f}s, "
                f"quantum: {quantum_time_per_run:.6f}s, "
                f"post_process: {post_time_avg:.6f}s"
            )

    # ---- Plot 1: total time vs message length ----
    for scenario in scenarios:
        plt.plot(
            lam_ot_list,
            total_times[scenario],
            marker='o',
            label=f"{scenario.capitalize()} total",
        )

    plt.title("Total protocol time vs message length")
    plt.xlabel("Message length lam_OT (bits)")
    plt.ylabel("Average total time per run (s)")
    plt.grid()
    plt.legend()
    datetime = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"time_vs_message_length_total_{datetime}.pdf")
    plt.show()

    # ---- Plot 2: quantum vs post-processing contributions ----
    for scenario in scenarios:
        plt.plot(
            lam_ot_list,
            quantum_times[scenario],
            marker='o',
            label=f"{scenario.capitalize()} quantum",
        )
        plt.plot(
            lam_ot_list,
            post_times[scenario],
            marker='x',
            label=f"{scenario.capitalize()} post-processing",
        )

    plt.title("Quantum vs post-processing time vs message length")
    plt.xlabel("Message length lam_OT (bits)")
    plt.ylabel("Average time per run (s)")
    # set y axis to log scale to better see differences at smaller message lengths
    plt.yscale('log')
    plt.grid()
    plt.legend()
    datetime = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"time_components_vs_message_length_{datetime}.pdf")
    plt.show()

    # Save raw data for later analysis / LaTeX plots
    np.savez(
        f"time_vs_message_length_data_{datetime}.npz",
        lam_ot_list=lam_ot_list,
        total_times=total_times,
        quantum_times=quantum_times,
        post_times=post_times,
    )


def _run_single_time_until_succesful(leakage_rate, lam_ot, n):
    """
    Run the protocol n times for a given scenario and message length (lam_ot),
    and return average total time per run, split into quantum and post-processing parts.
    """
    # Load scenario-specific configs
    scenario = "optimistic"  # You can also test "current" scenario if needed
    cfg = StackNetworkConfig.from_file(f"./params/qdevice_params_{scenario}.yaml")
    depolarise_config = DepolariseLinkConfig.from_file(f"./params/link_params_{scenario}.yaml")
    link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
    cfg.links = [link]



    # Use LDPC + Carter-Wegman as specified
    error_correction = LDPCErrorCorrection()
    hash_function = CarterWegmanHash()

    succesful_total_time = []
    current_att_length = 0
    max_attempts = 1000  # To prevent infinite loops in case of very low success rates
    attempts = 0
    messages = ["hi, i'm alice, encoding a message", "you might see me, or the other message, i don't know which one"]
    while len(succesful_total_time) <= n and attempts < max_attempts:
        alice_program = AliceProgram(
            messages=messages,
            lam_OT=lam_ot,
            lam_HS=lam_HS,
            Q_tol=Q_tol,
            lam_PQS=lam_PQS,
            error_correction=error_correction,
            hash_function=hash_function,
            leakage_rate=leakage_rate,
            # you can add leakage_rate here if needed, but for timing it's not required
        )
        bob_program = BobProgram(
            lam_OT=lam_ot,
            lam_HS=lam_HS,
            lam_PQS=lam_PQS,
            chosen_message=chosen_message,
            error_correction=error_correction,
            hash_function=hash_function,
            leakage_rate=leakage_rate,
        )

        # Run with statistics enabled
        stats, res = run(
            config=cfg,
            programs={"Alice": alice_program, "Bob": bob_program},
            num_times=1,
            measure_stats=True,
        )
        res_alice, res_bob = res

        # elapsed_sim_time is in ns (comment in your code), convert to seconds
        sim_time_total_s = stats.data["elapsed_sim_time"] * 1e-9
        quantum_time_per_run = sim_time_total_s

        # Bob's post-processing time is per run and already in seconds
        post_times = res_bob[0]["post_process_time"]
        post_time_avg = post_times

        total_time_avg = quantum_time_per_run + post_time_avg
        current_att_length += total_time_avg    
        attempts += 1
        print(
            f"Attempt {attempts}, Leakage Rate: {leakage_rate}, lam_OT: {lam_ot}, n: {len(succesful_total_time)}, ")

        # check if succesful
        if res_alice[0]['m0'] == res_bob[0]['m0'] or res_alice[0]['m1'] == res_bob[0]['m1']:
            succesful_total_time.append(current_att_length)
            current_att_length = 0  # reset for next successful run
            attempts = 0  # reset attempts for next successful run

    return succesful_total_time


def time_until_succesful():
    # Message lengths to test (in bits). Adjust this list as needed.
    # message_lengths = np.linspace(128, 16384, num=6, dtype=int)
    message_lengths = np.linspace(128, 7000, num=18, dtype=int)

    # Number of protocol executions per data point (per scenario, per lam_OT)
    n = 20

    # leakage_rates = [0.1, 0.2, 0.3]
    leakage_rates = [0.4, 0.35, 0.3]

    # Store results: average times per run
    total_times = {str(rate): [] for rate in leakage_rates}
    error_times = {str(rate): [] for rate in leakage_rates}
    # quantum_times = {rate: [] for rate in leakage_rates}
    # post_times = {rate: [] for rate in leakage_rates}

    for rate in leakage_rates:
        results = []

        # Parallelise over message lengths for this scenario
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_run_single_time_until_succesful, rate, length, n): length
                for length in message_lengths
            }

            for future in as_completed(futures):
                succesful_total_time = future.result()
                average_time = np.mean(succesful_total_time)
                errror_bar = np.std(succesful_total_time) / np.sqrt(len(succesful_total_time))
                results.append((average_time, errror_bar))

        # Sort results by message length to align with lam_ot_list
        results.sort(key=lambda t: t[0])

        for succesful_total_time, error in results:
            total_times[str(rate)].append(succesful_total_time)
            error_times[str(rate)].append(error)

    # ---- Plot 1: total time vs message length ----
    print("Total times for each leakage rate:")
    for rate in leakage_rates:
        print(f"  {rate}: {total_times[str(rate)]}")



    plt.title("Total time taken until successful OT for different $\\lambda_{OT}$")
    plt.xlabel("$\\lambda_{OT}$")
    plt.ylabel("Average total time per run (minutes)")

    for rate in leakage_rates:
        # plt.plot(
        #     message_lengths,
        #     total_times[str(rate)],
        #     yerr=error_times[str(rate)],
        #     marker='o',
        #     label=f"{rate} total",
        # )
        # plot with error bars
        plt.errorbar(
            message_lengths,
            # convert to minutes for better readability
            np.array(total_times[str(rate)]) / 60,
            yerr=np.array(error_times[str(rate)]) / 60,
            marker='o',
            label=f"leakage rate of {rate}",
        )
    plt.grid()
    plt.legend()
    datetime = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"time_vs_message_length_total_succesful_{datetime}.pdf")

    # plot again but with log scale on y axis to better see differences at smaller message lengths
    # first clear the current plot
    plt.clf()

    plt.title("Total time taken until successful OT for different $\\lambda_{OT}$")
    plt.xlabel("$\\lambda_{OT}$ (bits)")
    plt.ylabel("Average total time per run (minutes)")

    for rate in leakage_rates:
        # plt.plot(
        #     message_lengths,
        #     total_times[str(rate)],
        #     yerr=error_times[str(rate)],
        #     marker='o',
        #     label=f"{rate} total",
        # )
        # plot with error bars
        plt.errorbar(
            message_lengths,
            # convert to minutes for better readability
            np.array(total_times[str(rate)]) / 60,
            yerr=np.array(error_times[str(rate)]) / 60,
            marker='o',
            label=f"leakage rate of {rate}",
        )
    plt.grid()
    plt.legend()
    # log scale for y axis to better see differences at smaller message lengths
    plt.yscale('log')
    datetime = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"time_vs_message_length_total_succesful_{datetime}_log.pdf")
    # save data
    np.savez(
        f"time_vs_message_length_total_succesful_data_{datetime}.npz",
        message_lengths=message_lengths,
        total_times=total_times,
        error_times=error_times,
    )

    # now plot again on the log scale, but the y axis show the time devided by the message length, to see how the time scales with the message length
    plt.clf()
    plt.title("Total time taken until successful OT per bit for different $\\lambda_{OT}$")
    plt.xlabel("$\\lambda_{OT}$ (bits)")
    plt.ylabel("Average total time per bit (seconds)")
    for rate in leakage_rates:
        # plot with error bars
        plt.errorbar(
            message_lengths,
            # convert to seconds and divide by message length to get time per bit
            np.array(total_times[str(rate)]) / message_lengths,
            yerr=np.array(error_times[str(rate)]) / message_lengths,
            marker='o',
            label=f"leakage rate of {rate}",
        )
    plt.grid()
    plt.legend()
    datetime = time.strftime("%Y%m%d-%H%M%S")
    # set the width of the saved figure to be larger to better see the differences at smaller message lengths
    plt.savefig(f"time_per_bit_vs_message_length_total_succesful_{datetime}_log.pdf")
    


    plt.show()


if __name__ == "__main__":
    # calculate_qber_for_fidelity()
    # calculate_minimum_leakage_rate()
    # calculate_leakage_succes_rate()
    # calculate_time_taken()
    # time_until_succesful()
    pass



# import network configuration from file
n = 1
messages = ["Hello, Bob!", "This is a secret message for Bob."]
cfg = StackNetworkConfig.from_file("./params/qdevice_params_optimistic.yaml")
depolarise_config = DepolariseLinkConfig.from_file("./params/link_params_optimistic.yaml")
link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
# # # Replace link from YAML file with new depolarise link
cfg.links = [link]
alice_program = AliceProgram(messages=messages, lam_OT=lam_OT, lam_HS=lam_HS, Q_tol=Q_tol, lam_PQS=lam_PQS, error_correction=error_correction, hash_function=hash_function, leakage_rate=0.2)
bob_program = BobProgram(lam_OT=lam_OT, lam_HS=lam_HS, lam_PQS=lam_PQS, chosen_message=chosen_message, error_correction=error_correction, hash_function=hash_function, leakage_rate=0.2)
stats, res = run(config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=n, measure_stats=True)
res_alice, res_bob = res
# get avg qber
avg_qber = np.mean([r['qber'] for r in res_alice])
print(f"Average QBER: {avg_qber}")
# display all qber values
for i, r in enumerate(res_alice):
    print(f"Run {i}: QBER = {r['qber']}, m0 = {r['m0']}, m1 = {r['m1']}")
print(stats.data['elapsed_wall_time'], stats.data['elapsed_sim_time'])
post_process_time = res_bob[0]['post_process_time']
print(res_alice, res_bob, post_process_time)