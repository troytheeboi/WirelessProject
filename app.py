import math
from math import log10, log2
import pandas as pd
from flask import Flask, render_template, request, flash
import os

app = Flask(__name__)
app.secret_key = 'hicbxzklschcidhshxbhsxnsxhxks'


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


# Lookup tables (example values)
bpsk_qpsk = {
    10 ** -1: 0, 10 ** -2: 4, 10 ** -3: 7, 10 ** -4: 8, 10 ** -5: 9.6, 10 ** -6: 10.5, 10 ** -7: 11.2, 10 ** -8: 12
}
eight_psk = {
    10 ** -1: 0, 10 ** -2: 7, 10 ** -3: 10, 10 ** -4: 12, 10 ** -5: 13, 10 ** -6: 14, 10 ** -7: 14.8, 10 ** -8: 15.7
}
sixteen_psk = {
    10 ** -1: 0, 10 ** -2: 11, 10 ** -3: 14, 10 ** -4: 16, 10 ** -5: 17, 10 ** -6: 18.8, 10 ** -7: 20, 10 ** -8: 19
}


def lookup_ebn0(ber, scheme):
    if scheme == "BPSK/QPSK":
        table = bpsk_qpsk
    elif scheme == "8-PSK":
        table = eight_psk
    elif scheme == "16-PSK":
        table = sixteen_psk
    else:
        raise ValueError("Invalid modulation scheme")

    # Find closest BER
    closest_ber = min(table.keys(), key=lambda k: abs(k - ber))
    return table[closest_ber]


@app.route('/')
def question1():
    return render_template('question1.html')


@app.route('/questionOne')
def questionOne():
    return render_template('question1.html')


@app.route('/question2')
def question2():
    return render_template('question2.html')


@app.route('/question3')
def question3():
    return render_template('question3.html')


@app.route('/question4')
def question4():
    return render_template('question4.html')


@app.route('/question5')
def question5():
    return render_template('question5.html')


@app.route('/question1', methods=['POST'])
def calculate():
    goOn = True

    try:
        bandwidth = request.form['bandwidth']
        quantizer_bits = request.form['quantizer_bits']
        source_encoder_rate = request.form['source_encoder_rate']
        channel_encoder_rate = request.form['channel_encoder_rate']
        interleaver_bits = request.form['interleaver_bits']

        # Check if all inputs are numbers
        if not all(
                map(lambda x: x.replace('.', '', 1).isdigit(),
                    [bandwidth, source_encoder_rate, channel_encoder_rate])):
            flash("Bandwidth, Source Encoder Compression Rate, and Channel Encoder Compression Rate must be numbers",
                  "error")
            goOn = False

        # Check if quantizer bits and interleaver bits are integers
        if not quantizer_bits.isdigit():
            flash("Quantizer Bits must be an integer", "error")
            goOn = False

        if not interleaver_bits.isdigit():
            flash("Interleaver Bits must be an integer", "error")
            goOn = False

        bandwidth = float(bandwidth)
        quantizer_bits = int(quantizer_bits)
        source_encoder_rate = float(source_encoder_rate)
        channel_encoder_rate = float(channel_encoder_rate)
        interleaver_bits = int(interleaver_bits)

        sampling_frequency = calculate_sampling_frequency(bandwidth)
        quantizer_levels = calculate_quantizer_levels(quantizer_bits)
        source_encoder_output_rate = calculate_encoder_output_rate(sampling_frequency, source_encoder_rate,
                                                                   quantizer_bits)
        channel_encoder_output_rate = calculate_channel_encoder_output_rate(source_encoder_output_rate,
                                                                            channel_encoder_rate)
        interleaver_output_rate = calculate_interleaver_output_rate(channel_encoder_output_rate)

        if (goOn):
            return render_template('question1.html', sampling_frequency=sampling_frequency,
                                   quantizer_levels=quantizer_levels,
                                   source_encoder_output_rate=source_encoder_output_rate,
                                   channel_encoder_output_rate=channel_encoder_output_rate,
                                   interleaver_output_rate=interleaver_output_rate,
                                   bandwidth=bandwidth,
                                   quantizer_bits=quantizer_bits,
                                   source_encoder_rate=source_encoder_rate,
                                   channel_encoder_rate=channel_encoder_rate,
                                   interleaver_bits=interleaver_bits)
        else:
            return render_template('question1.html',
                                   bandwidth=request.form['bandwidth'],
                                   quantizer_bits=request.form['quantizer_bits'],
                                   source_encoder_rate=request.form['source_encoder_rate'],
                                   channel_encoder_rate=request.form['channel_encoder_rate'],
                                   interleaver_bits=request.form['interleaver_bits'])

    except Exception as e:
        return render_template('question1.html',
                               bandwidth=request.form['bandwidth'],
                               quantizer_bits=request.form['quantizer_bits'],
                               source_encoder_rate=request.form['source_encoder_rate'],
                               channel_encoder_rate=request.form['channel_encoder_rate'],
                               interleaver_bits=request.form['interleaver_bits'])


def calculate_sampling_frequency(bandwidth):
    # Your calculation logic here
    return bandwidth * 2


def calculate_quantizer_levels(quantizer_bits):
    # Your calculation logic here
    return pow(2, quantizer_bits)


def calculate_encoder_output_rate(sampling_freq, compression_rate, quantizer_bits):
    # Your calculation logic here
    return sampling_freq * quantizer_bits * compression_rate


def calculate_channel_encoder_output_rate(encoder_rate, compression_rate):
    # Your calculation logic here
    return encoder_rate / compression_rate


def calculate_interleaver_output_rate(interleaver_bits):
    # Your calculation logic here
    return interleaver_bits


@app.route('/questionTwo', methods=['POST'])
def questiontwo():
    goOn2 = True

    try:
        # Get form data
        bandwidth = request.form['bandwidth']
        subcarrier_spacing = request.form['subcarrier_spacing']
        ofdm_symbols = request.form['ofdm_symbols']
        rb_duration = request.form['rb_duration']
        qam_bits = request.form['qam_bits']
        num_parallel_rbs = request.form['num_parallel_rbs']

        # Validate inputs
        if not all(map(lambda x: x.replace('.', '', 1).replace('-', '', 1).isdigit(),
                       [bandwidth, subcarrier_spacing, ofdm_symbols, rb_duration, qam_bits, num_parallel_rbs])):
            flash("All fields have to be numbers", "error")
            goOn2 = False

        if not ofdm_symbols.isdigit():
            flash("Number of Symbols has to be an integer", "error")
            goOn2 = False

        if not qam_bits.isdigit():
            flash("Number of QAM bits has to be an integer", "error")
            goOn2 = False

        if not num_parallel_rbs.isdigit():
            flash("Number of parallel Resource Blocks has to be an integer", "error")
            goOn2 = False

        if not log2(int(qam_bits)).is_integer():
            flash("QAM bit number has to be from 2^X", "error")
            goOn2 = False

        if not (float(bandwidth) % float(subcarrier_spacing)) == float(0):
            flash("Bandwidth has to be divisible by Spacing", "error")
            goOn2 = False
            raise Exception("")

        # Convert inputs to appropriate types
        bandwidth = float(bandwidth)
        subcarrier_spacing = float(subcarrier_spacing)
        ofdm_symbols = int(ofdm_symbols)
        rb_duration = float(rb_duration)
        qam_bits = int(qam_bits)
        num_parallel_rbs = int(num_parallel_rbs)

        # Perform calculations
        bits_per_resource_element = calculate_bits_per_resource_element(qam_bits)
        bits_per_ofdm_symbol = calculate_bits_per_ofdm_symbol(bits_per_resource_element, bandwidth, subcarrier_spacing)
        bits_per_ofdm_rb = calculate_bits_per_ofdm_rb(bits_per_ofdm_symbol, ofdm_symbols)
        max_transmission_rate = calculate_max_transmission_rate(bits_per_ofdm_rb, num_parallel_rbs, rb_duration)

        if goOn2:
            return render_template('question2.html',
                                   bits_per_resource_element=bits_per_resource_element,
                                   bits_per_ofdm_symbol=bits_per_ofdm_symbol,
                                   bits_per_ofdm_rb=bits_per_ofdm_rb,
                                   max_transmission_rate=max_transmission_rate,
                                   bandwidth=bandwidth,
                                   subcarrier_spacing=subcarrier_spacing,
                                   ofdm_symbols=ofdm_symbols,
                                   rb_duration=rb_duration,
                                   qam_bits=qam_bits,
                                   num_parallel_rbs=num_parallel_rbs)
        else:
            return render_template('question2.html',
                                   bandwidth=request.form['bandwidth'],
                                   subcarrier_spacing=request.form['subcarrier_spacing'],
                                   ofdm_symbols=request.form['ofdm_symbols'],
                                   rb_duration=request.form['rb_duration'],
                                   qam_bits=request.form['qam_bits'],
                                   num_parallel_rbs=request.form['num_parallel_rbs'])
    except Exception as e:
        return render_template('question2.html',
                               bandwidth=request.form['bandwidth'],
                               subcarrier_spacing=request.form['subcarrier_spacing'],
                               ofdm_symbols=request.form['ofdm_symbols'],
                               rb_duration=request.form['rb_duration'],
                               qam_bits=request.form['qam_bits'],
                               num_parallel_rbs=request.form['num_parallel_rbs'])


def calculate_bits_per_resource_element(qam_bits):
    return int(log2(qam_bits))


def calculate_bits_per_ofdm_symbol(bits_per_resource_element, bandwidth, subcarrier_spacing):
    num_subcarriers = bandwidth / subcarrier_spacing
    return int(bits_per_resource_element * num_subcarriers)


def calculate_bits_per_ofdm_rb(bits_per_ofdm_symbol, ofdm_symbols):
    return int(bits_per_ofdm_symbol * ofdm_symbols)


def calculate_max_transmission_rate(bits_per_ofdm_rb, num_parallel_rbs, rb_duration):
    return (bits_per_ofdm_rb * num_parallel_rbs) / (rb_duration)  # converting millisec to sec


@app.route('/questionThree', methods=['POST'])
def questionthree():
    goOn2 = True

    try:

        path_loss = request.form['path_loss']
        frequency = request.form['frequency']
        transmit_antenna_gain = request.form['transmit_antenna_gain']
        receive_antenna_gain = request.form['receive_antenna_gain']
        data_rate = request.form['data_rate']
        antenna_feed_loss = request.form['antenna_feed_loss']
        other_losses = request.form['other_losses']
        fade_margin = request.form['fade_margin']
        receiver_amplifier_gain = request.form['receiver_amplifier_gain']
        total_noise_figure = request.form['total_noise_figure']
        noise_temperature = request.form['noise_temperature']
        link_margin = request.form['link_margin']
        max_bit_error_rate = request.form['max_bit_error_rate']
        output_power_unit = request.form['output_power_unit']
        output_modulation_scheme = request.form['modulation_scheme']

        # # Validate inputs
        # if not all(map(lambda x: x.replace('.', '', 1).replace('-', '', 1).isdigit(),
        #                [path_loss, frequency, transmit_antenna_gain, receive_antenna_gain, data_rate, antenna_feed_loss,
        #                 other_losses, fade_margin,
        #                 receiver_amplifier_gain, total_noise_figure, noise_temperature, link_margin, max_bit_error_rate,
        #                 ])):
        #     flash("All fields have to be numbers", "error")
        #     goOn2 = False

        if not isfloat(path_loss):
            flash("Path Loss has to be a number", "error")
            goOn2 = False

        if not isfloat(frequency):
            flash("Frequency has to be a number", "error")
            goOn2 = False

        if not isfloat(transmit_antenna_gain):
            flash("Transmitter Gain has to be a number", "error")
            goOn2 = False

        if not isfloat(receive_antenna_gain):
            flash("Receiver Gain has to be a number", "error")
            goOn2 = False

        if not isfloat(data_rate):
            flash("Data Rate has to be a number", "error")
            goOn2 = False

        if not isfloat(antenna_feed_loss):
            flash("Feed Loss has to be a number", "error")
            goOn2 = False

        if not isfloat(other_losses):
            flash("Other Losses has to be a number", "error")
            goOn2 = False

        if not isfloat(fade_margin):
            flash("Fade Margin has to be a number", "error")
            goOn2 = False

        if not isfloat(receiver_amplifier_gain):
            flash("Receiver Amp Gain has to be a number", "error")
            goOn2 = False

        if not isfloat(total_noise_figure):
            flash("Total Noise Figure has to be a number", "error")
            goOn2 = False

        if not isfloat(noise_temperature):
            flash("Noise Temp has to be a number", "error")
            goOn2 = False

        if not isfloat(link_margin):
            flash("Link Margin has to be a number", "error")
            goOn2 = False

        if not isfloat(max_bit_error_rate):
            flash("BER has to be a number", "error")
            goOn2 = False

        path_loss = float(request.form['path_loss'])
        frequency = float(request.form['frequency'])
        transmit_antenna_gain = float(request.form['transmit_antenna_gain'])
        receive_antenna_gain = float(request.form['receive_antenna_gain'])
        data_rate = float(request.form['data_rate'])
        antenna_feed_loss = float(request.form['antenna_feed_loss'])
        other_losses = float(request.form['other_losses'])
        fade_margin = float(request.form['fade_margin'])
        receiver_amplifier_gain = float(request.form['receiver_amplifier_gain'])
        total_noise_figure = float(request.form['total_noise_figure'])
        noise_temperature = float(request.form['noise_temperature'])
        link_margin = float(request.form['link_margin'])
        max_bit_error_rate = float(request.form['max_bit_error_rate'])
        output_power_unit = request.form['output_power_unit']
        output_modulation_scheme = request.form['modulation_scheme']

        power_unit_path_loss = request.form['power_unit_path_loss']
        power_unit_frequency = request.form['power_unit_frequency']
        power_unit_transmit_antenna_gain = request.form['power_unit_transmit_antenna_gain']
        power_unit_receive_antenna_gain = request.form['power_unit_receive_antenna_gain']
        power_unit_data_rate = request.form['power_unit_data_rate']
        power_unit_antenna_feed_loss = request.form['power_unit_antenna_feed_loss']
        power_unit_other_losses = request.form['power_unit_other_losses']
        power_unit_fade_margin = request.form['power_unit_fade_margin']
        power_unit_receiver_amplifier_gain = request.form['power_unit_receiver_amplifier_gain']
        power_unit_total_noise_figure = request.form['power_unit_total_noise_figure']
        power_unit_noise_temperature = request.form['power_unit_noise_temperature']
        power_unit_link_margin = request.form['power_unit_link_margin']

        # Placeholder for the required transmit power calculation
        # You'll need to replace this with the actual calculation logic
        required_transmit_power = calculate_required_transmit_power(
            path_loss, frequency, transmit_antenna_gain,
            receive_antenna_gain, data_rate, antenna_feed_loss,
            other_losses, fade_margin, receiver_amplifier_gain,
            total_noise_figure, noise_temperature, link_margin,
            max_bit_error_rate, output_power_unit, output_modulation_scheme, power_unit_path_loss,
            power_unit_frequency, power_unit_transmit_antenna_gain,
            power_unit_receive_antenna_gain, power_unit_data_rate,
            power_unit_antenna_feed_loss, power_unit_other_losses, power_unit_fade_margin,
            power_unit_receiver_amplifier_gain,
            power_unit_total_noise_figure, power_unit_noise_temperature,
            power_unit_link_margin
        )

        required_transmit_power_watt = db_to_watt(required_transmit_power)

        if goOn2:
            return render_template('question3.html', required_transmit_power=required_transmit_power,
                                   required_transmit_power_watt=required_transmit_power_watt,
                                   path_loss=path_loss, frequency=frequency,
                                   transmit_antenna_gain=transmit_antenna_gain,
                                   receive_antenna_gain=receive_antenna_gain,
                                   data_rate=data_rate, antenna_feed_loss=antenna_feed_loss,
                                   other_losses=other_losses, fade_margin=fade_margin,
                                   receiver_amplifier_gain=receiver_amplifier_gain,
                                   total_noise_figure=total_noise_figure,
                                   noise_temperature=noise_temperature, link_margin=link_margin,
                                   max_bit_error_rate=max_bit_error_rate, power_unit=output_power_unit,
                                   modulation_scheme=output_modulation_scheme)

    except ValueError as e:
        return render_template('question3.html', path_loss=path_loss, frequency=frequency,
                               transmit_antenna_gain=transmit_antenna_gain,
                               receive_antenna_gain=receive_antenna_gain,
                               data_rate=data_rate, antenna_feed_loss=antenna_feed_loss,
                               other_losses=other_losses, fade_margin=fade_margin,
                               receiver_amplifier_gain=receiver_amplifier_gain,
                               total_noise_figure=total_noise_figure,
                               noise_temperature=noise_temperature, link_margin=link_margin,
                               max_bit_error_rate=max_bit_error_rate, power_unit=output_power_unit,
                               modulation_scheme=output_modulation_scheme)


def watt_to_db(watt):
    """Convert Watt to dB."""
    return 10 * log10(watt)


def db_to_watt(db):
    """Convert dB to Watt."""
    return 10 ** (db / 10)


def calculate_required_transmit_power(path_loss, frequency, transmit_antenna_gain,
                                      receive_antenna_gain, data_rate, antenna_feed_loss,
                                      other_losses, fade_margin, receiver_amplifier_gain,
                                      total_noise_figure, noise_temperature, link_margin,
                                      max_bit_error_rate, output_power_unit, modulation_scheme, power_unit_path_loss,
                                      power_unit_frequency, power_unit_transmit_antenna_gain,
                                      power_unit_receive_antenna_gain, power_unit_data_rate,
                                      power_unit_antenna_feed_loss, power_unit_other_losses, power_unit_fade_margin,
                                      power_unit_receiver_amplifier_gain,
                                      power_unit_total_noise_figure, power_unit_noise_temperature,
                                      power_unit_link_margin):
    frequency = frequency * 10 ** 6
    data_rate = data_rate * 1000

    if (power_unit_path_loss == 'Watt'):
        path_loss = watt_to_db(path_loss)
    if (power_unit_frequency == 'Hz'):
        frequency = watt_to_db(frequency)
    if (power_unit_transmit_antenna_gain == 'Watt'):
        transmit_antenna_gain = watt_to_db(transmit_antenna_gain)
    if (power_unit_receive_antenna_gain == 'Watt'):
        receive_antenna_gain = watt_to_db(receive_antenna_gain)
    if (power_unit_data_rate == 'kbps'):
        data_rate = watt_to_db(data_rate)
    if (power_unit_antenna_feed_loss == 'Watt'):
        antenna_feed_loss = watt_to_db(antenna_feed_loss)
    if (power_unit_other_losses == 'Watt'):
        other_losses = watt_to_db(other_losses)
    if (power_unit_fade_margin == 'Watt'):
        fade_margin = watt_to_db(fade_margin)
    if (power_unit_receiver_amplifier_gain == 'Watt'):
        receiver_amplifier_gain = watt_to_db(receiver_amplifier_gain)
    if (power_unit_total_noise_figure == 'Watt'):
        total_noise_figure = watt_to_db(total_noise_figure)
    if (power_unit_noise_temperature == 'K'):
        noise_temperature = watt_to_db(noise_temperature)
    if (power_unit_link_margin == 'Watt'):
        link_margin = watt_to_db(link_margin)

    pr_dB = link_margin - 228.6 + noise_temperature + total_noise_figure + data_rate + lookup_ebn0(max_bit_error_rate,
                                                                                                   modulation_scheme)
    pt_db = pr_dB + path_loss + antenna_feed_loss + other_losses + fade_margin - transmit_antenna_gain - receive_antenna_gain - receiver_amplifier_gain

    return pt_db


@app.route('/questionFour', methods=['POST'])
def questionfour():
    goOn2 = True

    try:
        # Retrieve input values from form
        transmission_bandwidth = request.form['transmission_bandwidth']
        propagation_time = request.form['propagation_time']
        frame_size = request.form['frame_size']
        frame_rate = request.form['frame_rate']

        # # Validate inputs
        # if not all(map(lambda x: x.replace('.', '', 1).replace('-', '', 1).isdigit(),
        #                [transmission_bandwidth, propagation_time, frame_size, frame_rate])):
        #     flash("All fields are required", "error")
        #     goOn2 = False

        if not isfloat(transmission_bandwidth):
            flash("Transmission Bandwidth has to be a number", "error")
            goOn2 = False

        if not isfloat(propagation_time):
            flash("Propagation Time has to be a number", "error")
            goOn2 = False

        if not isfloat(frame_size):
            flash("Frame Size has to be a number", "error")
            goOn2 = False

        if not isfloat(frame_rate):
            flash("Frame Rate has to be a number", "error")
            goOn2 = False

        bw_bfr = transmission_bandwidth
        prop_bfr = propagation_time
        fram_size_bfr = frame_size
        fram_rate_bfr = frame_rate

        if (goOn2):
            # Retrieve input values from form
            transmission_bandwidth = float(request.form['transmission_bandwidth'])
            propagation_time = float(request.form['propagation_time'])
            frame_size = float(request.form['frame_size'])
            frame_rate = float(request.form['frame_rate'])

            transmission_bandwidth = transmission_bandwidth * 10 ** 6
            propagation_time = propagation_time * 10 ** -6
            frame_size = frame_size * 1000
            frame_rate = frame_rate * 1000

            # Calculate throughputs
            throughput_pure_aloha = calculate_throughput_pure_aloha(transmission_bandwidth, frame_size, frame_rate)

            throughput_slotted_aloha = calculate_throughput_slotted_aloha(transmission_bandwidth, frame_size,
                                                                          frame_rate)

            throughput_unslotted_nonpersistent_csma = calculate_throughput_unslotted_nonpersistent_csma(
                transmission_bandwidth, propagation_time, frame_size, frame_rate)

            throughput_slotted_nonpersistent_csma = calculate_throughput_slotted_nonpersistent_csma(
                transmission_bandwidth,
                propagation_time,
                frame_size, frame_rate)

            throughput_slotted_persistent_csma = calculate_throughput_slotted_persistent_csma(transmission_bandwidth,
                                                                                              propagation_time,
                                                                                              frame_size,
                                                                                              frame_rate)

            throughput_pure_aloha = throughput_pure_aloha * 100
            throughput_slotted_aloha = throughput_slotted_aloha * 100
            throughput_unslotted_nonpersistent_csma = throughput_unslotted_nonpersistent_csma * 100
            throughput_slotted_nonpersistent_csma = throughput_slotted_nonpersistent_csma * 100
            throughput_slotted_persistent_csma = throughput_slotted_persistent_csma * 100

            return render_template('question4.html',
                                   transmission_bandwidth=bw_bfr,
                                   propagation_time=prop_bfr,
                                   frame_size=fram_size_bfr,
                                   frame_rate=fram_rate_bfr,
                                   throughput_pure_aloha=throughput_pure_aloha,
                                   throughput_slotted_aloha=throughput_slotted_aloha,
                                   throughput_unslotted_nonpersistent_csma=throughput_unslotted_nonpersistent_csma,
                                   throughput_slotted_nonpersistent_csma=throughput_slotted_nonpersistent_csma,
                                   throughput_slotted_persistent_csma=throughput_slotted_persistent_csma)

        else:
            return render_template('question4.html',
                                   transmission_bandwidth=bw_bfr,
                                   propagation_time=prop_bfr,
                                   frame_size=fram_size_bfr,
                                   frame_rate=fram_rate_bfr)

    except ValueError as e:
        return render_template('question4.html',
                               transmission_bandwidth=bw_bfr,
                               propagation_time=prop_bfr,
                               frame_size=fram_size_bfr,
                               frame_rate=fram_rate_bfr)


def calculate_throughput_pure_aloha(transmission_bandwidth, frame_size, frame_rate):
    G = frame_rate * frame_size / transmission_bandwidth
    return (G * math.exp(-2 * G))


def calculate_throughput_slotted_aloha(transmission_bandwidth, frame_size, frame_rate):
    G = frame_rate * frame_size / transmission_bandwidth
    return (G * math.exp(-G))


def calculate_throughput_unslotted_nonpersistent_csma(transmission_bandwidth, propagation_time, frame_size, frame_rate):
    T = (frame_size / transmission_bandwidth)
    G = frame_rate * T
    a = propagation_time / T

    return ((G * math.exp(-2 * a * T)) / (G * (1 + 2 * a) + math.exp(-a * G)))


def calculate_throughput_slotted_nonpersistent_csma(transmission_bandwidth, propagation_time, frame_size, frame_rate):
    T = (frame_size / transmission_bandwidth)
    G = frame_rate * T
    a = propagation_time / T

    return ((a * G * math.exp(-2 * a * T)) / (1 - math.exp(-a * G) + a))


def calculate_throughput_slotted_persistent_csma(transmission_bandwidth, propagation_time, frame_size, frame_rate):
    T = (frame_size / transmission_bandwidth)
    G = frame_rate * T
    a = propagation_time / T

    return ((G * (1 + a - math.exp(-a * G)) * math.exp(-G * (1 + a))) / (
            (1 + a) * (1 - math.exp(-a * G)) + a * math.exp(-G * (1 + a))))


@app.route('/question5', methods=['POST'])
def questionfive():

    goOn2 = True
    try:

        timeslots_per_carrier = request.form['timeslots_per_carrier']
        area = request.form['area']
        num_subscribers = request.form['num_subscribers']
        calls_per_day = request.form['calls_per_day']
        avg_call_duration = request.form['avg_call_duration']
        grade_of_service = request.form['grade_of_service']
        min_sir_value = request.form['min_sir_value']
        min_sir_unit = request.form['min_sir_unit']
        power_reference_value = request.form['power_reference_value']
        power_reference_unit = request.form['power_reference_unit']
        reference_distance = request.form['reference_distance']
        path_loss_exponent = request.form['path_loss_exponent']
        receiver_sensitivity_value = request.form['receiver_sensitivity_value']
        receiver_sensitivity_unit = request.form['receiver_sensitivity_unit']

        # # Validate inputs
        # if not all(map(lambda x: (x.replace('.', '', 1).replace('-', '', 1)).isdigit(),
        #                [timeslots_per_carrier, area, num_subscribers, calls_per_day, avg_call_duration,
        #                 grade_of_service, min_sir_value, power_reference_value,
        #                 reference_distance, path_loss_exponent, receiver_sensitivity_value])):
        #     flash("All fields are required", "error")
        #     goOn2 = False

        if not timeslots_per_carrier.isdigit():
            flash("Timeslots has to be an integer", "error")
            goOn2 = False

        if not isfloat(area):
            flash("Area has to be a number", "error")
            goOn2 = False

        if not num_subscribers.isdigit():
            flash("Number of Subscribers has to be an integer", "error")
            goOn2 = False

        if not isfloat(calls_per_day):
            flash("Calls per Day has to be a number", "error")
            goOn2 = False

        if not isfloat(avg_call_duration):
            flash("Avg Call Duration has to be a number", "error")
            goOn2 = False

        if not isfloat(grade_of_service):
            flash("Grade of Service has to be a number", "error")
            goOn2 = False

        if not isfloat(min_sir_value):
            flash("Min SIR has to be a number", "error")
            goOn2 = False

        if not isfloat(power_reference_value):
            flash("Power Reference has to be a number", "error")
            goOn2 = False

        if not isfloat(reference_distance):
            flash("Reference Distance has to be a number", "error")
            goOn2 = False

        if not isfloat(path_loss_exponent):
            flash("Path Loss Exponent has to be a number", "error")
            goOn2 = False

        if not isfloat(receiver_sensitivity_value):
            flash("Receiver Sensitivity has to be a number", "error")
            goOn2 = False

        area_bfr = area
        min_sir_value_bfr = min_sir_value
        power_reference_value_bfr = power_reference_value
        sensitivity_bfr = receiver_sensitivity_value

        if (goOn2):

            timeslots_per_carrier = int(request.form['timeslots_per_carrier'])
            area = float(request.form['area'])
            num_subscribers = int(request.form['num_subscribers'])
            calls_per_day = float(request.form['calls_per_day'])
            avg_call_duration = float(request.form['avg_call_duration'])
            grade_of_service = float(request.form['grade_of_service'])
            min_sir_value = float(request.form['min_sir_value'])
            power_reference_value = float(request.form['power_reference_value'])
            reference_distance = float(request.form['reference_distance'])
            path_loss_exponent = float(request.form['path_loss_exponent'])
            receiver_sensitivity_value = float(request.form['receiver_sensitivity_value'])

            area_bfr = area
            area = area * 10 ** 6
            min_sir_value_bfr = 1
            min_sir_value_bfr = 1
            power_reference_value_bfr = 1

            if receiver_sensitivity_unit == 'μWatts':
                sensitivity_bfr = receiver_sensitivity_value
                receiver_sensitivity_value = receiver_sensitivity_value * 10 ** -6
            else:
                sensitivity_bfr = receiver_sensitivity_value
                receiver_sensitivity_value = db_to_watt(receiver_sensitivity_value)
                receiver_sensitivity_value = receiver_sensitivity_value * 10 ** -6

            if min_sir_unit == 'dB':
                min_sir_value_bfr = min_sir_value
                min_sir_value = db_to_watt(min_sir_value)

            if power_reference_unit == 'dB':
                power_reference_value_bfr = power_reference_value
                power_reference_value = db_to_watt(power_reference_value)

            # Calculate outputs
            max_transmitter_receiver_distance = calculate_max_distance(power_reference_value, reference_distance,
                                                                       path_loss_exponent, receiver_sensitivity_value)
            max_cell_size = calculate_max_cell_size(max_transmitter_receiver_distance)
            num_cells_service_area = calculate_num_cells_service_area(area, max_cell_size)
            total_traffic_load = calculate_total_traffic_load(num_subscribers, calls_per_day, avg_call_duration)
            traffic_load_per_cell = calculate_traffic_load_per_cell(total_traffic_load, num_cells_service_area)
            num_cells_per_cluster = calculate_num_cells_per_cluster(min_sir_value, path_loss_exponent)
            num_system_carriers = calculate_num_system_carriers(traffic_load_per_cell, grade_of_service,
                                                                timeslots_per_carrier, num_cells_per_cluster)
            # Render results
            return render_template('question5.html',
                                   timeslots_per_carrier=timeslots_per_carrier,
                                   area=area_bfr,
                                   num_subscribers=num_subscribers,
                                   calls_per_day=calls_per_day,
                                   avg_call_duration=avg_call_duration,
                                   grade_of_service=grade_of_service,
                                   min_sir_value=min_sir_value_bfr,
                                   min_sir_unit=min_sir_unit,
                                   power_reference_value=power_reference_value_bfr,
                                   power_reference_unit=power_reference_unit,
                                   reference_distance=reference_distance,
                                   path_loss_exponent=path_loss_exponent,
                                   receiver_sensitivity_value=sensitivity_bfr,
                                   receiver_sensitivity_unit=receiver_sensitivity_unit,
                                   max_transmitter_receiver_distance=max_transmitter_receiver_distance,
                                   max_cell_size=max_cell_size,
                                   num_cells_service_area=num_cells_service_area,
                                   total_traffic_load=total_traffic_load,
                                   traffic_load_per_cell=traffic_load_per_cell,
                                   num_cells_per_cluster=num_cells_per_cluster,
                                   num_system_carriers=num_system_carriers)
        else:
            return render_template('question5.html',
                                   timeslots_per_carrier=timeslots_per_carrier,
                                   area=area_bfr,
                                   num_subscribers=num_subscribers,
                                   calls_per_day=calls_per_day,
                                   avg_call_duration=avg_call_duration,
                                   grade_of_service=grade_of_service,
                                   min_sir_value=min_sir_value_bfr,
                                   min_sir_unit=min_sir_unit,
                                   power_reference_value=power_reference_value_bfr,
                                   power_reference_unit=power_reference_unit,
                                   reference_distance=reference_distance,
                                   path_loss_exponent=path_loss_exponent,
                                   receiver_sensitivity_value=sensitivity_bfr,
                                   receiver_sensitivity_unit=receiver_sensitivity_unit)
    except ValueError as e:
        return render_template('question5.html',
                               timeslots_per_carrier=timeslots_per_carrier,
                               area=area_bfr,
                               num_subscribers=num_subscribers,
                               calls_per_day=calls_per_day,
                               avg_call_duration=avg_call_duration,
                               grade_of_service=grade_of_service,
                               min_sir_value=min_sir_value_bfr,
                               min_sir_unit=min_sir_unit,
                               power_reference_value=power_reference_value_bfr,
                               power_reference_unit=power_reference_unit,
                               reference_distance=reference_distance,
                               path_loss_exponent=path_loss_exponent,
                               receiver_sensitivity_value=sensitivity_bfr,
                               receiver_sensitivity_unit=receiver_sensitivity_unit)


def calculate_max_distance(power_reference, reference_distance, path_loss_exponent, receiver_sensitivity):
    d = reference_distance / (receiver_sensitivity / power_reference) ** (1. / path_loss_exponent)

    return d


def calculate_max_cell_size(max_distance):
    return (3 * math.sqrt(3) / 2) * (max_distance ** 2)


def calculate_num_cells_service_area(area, max_cell_size):
    return math.ceil(area / max_cell_size)


def calculate_total_traffic_load(num_subscribers, calls_per_day, avg_call_duration):
    return num_subscribers * (calls_per_day / (24 * 60)) * avg_call_duration


def calculate_traffic_load_per_cell(total_traffic_load, num_cells_service_area):
    return total_traffic_load / num_cells_service_area


def calculate_num_cells_per_cluster(min_sir_value, loss_exponent):
    return math.ceil((((min_sir_value * 6) ** (1. / loss_exponent)) ** 2) / 3)


# Example usage
probabilities = [0.001, 0.002, 0.005, 0.010, 0.012, 0.013, 0.015, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200,
                 0.300, 0.400, 0.500]


def find_closest(probabilities, input_value):
    closest_value2 = min(probabilities, key=lambda x: abs(x - input_value))
    return closest_value2


# Function to find the (N) value where column value is closest to the target number
def find_closest_value(df, target_column, target_value):
    # Find the row index where the value in the target_column is closest to target_value
    closest_index = (df[target_column] - target_value).abs().idxmin()
    # Return the value in the (N) column for that row
    return df.loc[closest_index, '(N)']


def calculate_num_system_carriers(traffic_load_per_cell, grade_of_service, timeslots_per_carrier,num_cells_per_cluster):
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)

    # Construct the relative path to the Excel file
    file_path = os.path.join(current_dir, 'ErlangBTable.xlsx')

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    closest_prob = find_closest(probabilities, grade_of_service)

    N = find_closest_value(df, closest_prob, traffic_load_per_cell)

    return math.ceil(N / timeslots_per_carrier)*num_cells_per_cluster


if __name__ == '__main__':
    app.run(debug=True)
