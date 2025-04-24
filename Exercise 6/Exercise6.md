# MEDT4165 - Exercise 6

Written by Christoffer-Robin Vaksdal.

## Ultrasound imaging in general

1. There are many advantages to using ultrasound when compared to other
    imaging modalities. The main ones are: safety, cost-effectiveness and
    practicality.

    Unlike traditional x-ray imaging and CT-scanning,
    ultrasound does not use ionizing radiation which can cause DNA damage and
    contribute to the risk of cancer. This makes ultrasound a safer choice in
    general, but especially when it comes to diagnosing vulnerable/sensitive
    groups such as pregnant women or small children.

    Ultrasound imaging equipment is a lot cheaper to purchase, use and maintain
    than MRI- or CT-scanners.

    There exists a large range of ultrasound devices from big stationary systems
    to small handheld devices - most modern ultrasound imaging systems are at
    least somewhat portable (easily transportable on wheels within a hospital
    for example), making them especially useful for point-of-care imaging.

    Another great advantage is real-time imaging. MRI and CT typically produce
    static images (could be 2D or 3D) to be examined after the fact, while
    ultrasound gives a live feed of what is going one while the patient is
    still there. The real-time aspect also allows examinining movement such as
    flowing blood, heart beats or fetal movements.
2. An ultrasound wave is a high-frequency pressure wave or acoustic wave. That
    is to say, it is a longitudinal mechanical wave causing the medium it's
    propagating through to vary in pressure, density and particle velocity.
    Essentially it's a sound wave, except that as a general rule, humans are
    only able to discern sounds roughly between 20Hz and 20KHz. Ultrasound is a
    general term describing sound with a frequency too high to be heard by
    humans, so "high-frequency" in this case means >20KHz.
3. Ultrasound imaging typically uses sound in the range of a few MHz, but the
    specific frequency will wary depending on the use-case. The attenuation of
    sound through a medium is related to the frequency of the sound. In general,
    higher frequencies are attenuated more so for imaging deeper tissues one
    would use a lower frequency. There is a trade-off though, since frequency
    is also directly tied to axial/radial resolution (smaller SPL means better
    resolution).

    Cardiac or abdominal imaging typically use a frequency range of `~1MHz-5MHz`,
    while ophtalmic (eye-related) imaging might use frequencies as high as
    `15MHz-30MHz`.
4. The "pressure" in an ultrasound pulse can mean multiple things, but in
    general when it comes to medical ultrasound imaging what we care about is
    peak positive- and negative pressure caused by the wave as it propagates
    through the tissues in the body. The specific peak pressure undergone will
    depend on many things (frequency, pulse-duration, imaging mode, etc.), but
    is typically around `0.1MPa-3MPa`.
5. Scattering and specular reflection are two ways ultrasound waves interact
    with tissues. Scattering happens when the sound hits a small (compared to
    wavelength) or irregular structure and the wave is *scattered* in all
    directions. This causes weak echoes in many directions analogous to shining
    a light in thick fog. Scattering based imaging systems are not very
    angle-dependant and allows for characterization of different tissues, but
    you end up with unclear borders (low contrast). Scattering is useful for
    imaging organs like the liver or kidneys, or for examining muscle tissues.

    Specular reflection on the other hand happens when an ultrasound wave hits
    a large (relative to wavelength) smooth surface and is reflected back, like
    a mirror. This form of imaging gives clear distinct lines/borders, but is
    very dependant on the angle as the wave is only reflected back to the
    transducer if the surface is perpendicular to it. Specular reflection is
    useful for examining the walls of organs or the surface of bones.
6. Blood is made up mostly of plasma (a fluid, mostly water) and red blood
    cells. Red blood cells are very small compared to the wavelengths used in
    ultrasound imagery, meaning they scatter very weakly. Furthermore, blood is
    a liquid flowing through veins so it does not have any flat stationary
    surfaces to create specular reflections. The flow of the blood also further
    distorts any echoes coming from the blood making it even harder to make out.
7. The center frequency is the dominant frequency component of the ultrasound
    pulse, meaning the extent of any frequency-dependant effects on the signal
    will mostly depend on the center frequency. A shorter wavelength gives
    better axial resolution, but also increases attenuation meaning less
    penetration. The bandwith of the pulse is the range of frequencies around
    the center frequency that makes up the signal. A shorter pulse (in time)
    means a broader bandwith. Having a broader bandwidth improves axial
    resolution since your spatial pulse length is shorter. This makes it easier
    to differentiate tissues.
8. The axial/radial resolution is equal to half of the spatial pulse length
    (SPL). The formula is as follows: `SPL/2 ~= ( 𝜆N ) / 2 = (c * N) / (2 * f0)`
    Where `𝜆` is the wavelength, `N` is the number of periods in the pulse, `c`
    is the speed of sound and `f0` is the base/center frequency of the pulse.
9. To measure the pulse length you can use FWHM - Full Width Half Maximum. This
    is a measure of the *effective pulse length*, and it is equal to the width
    of the portion of the pulse envelope which is equal to or more than half of
    the amplitude. A quick way to get the envelope is to take the absolute value
    of the hilbert transform of your pulse.
10. Frequency dependant attenuation is the concept which describes how different
    frequencies are dampened or attenuated differently in a medium. Higher
    frequencies are attenuated more. This is a problem for ultrasound imaging
    where we often use higher frequencies to achieve higher resolution imagery.
    This has the drawback of the sound not being able to penetrate as far into
    the medium as it would at a lower frequency and leads to a trade-off between
    resolution and required depth when choosing a frequency to use.

## The ultrasound imaging system

11. The ultrasound imaging system front end is the part of the system right
    after the transducer. This front end is responsible for transmission and
    reception of ultrasound and typically consists of:
        - Receive amplifiers: 
        - A/D conversion: Measuring an analog signal and converting it to a digital form.
        - Beamforming: 
        - Demodulation and TGC: 
12. An ultrasound system channel is a 
13. The main parts of an ultrasound transducer stack are the backing layer,
    piezoelectric element(s), matching layer(s) and the acoustic lens.

    The backing layer is there to absorb vibrations moving backwards (the
    wrong way) from the transducer. This reduces unwanted echoes and can help
    increase axial resolution by improving the bandwith of the emitted pulse.

    The piezoelectric elements are the parts of the transducer that actually
    generate the ultrasound. Piezoelectricity is an effect where the electric
    charge and physical shape of a crystal become linked. A piezoelectric
    crystal will generate an electric charge in response to applied mechanical
    stress, and inversely it will physically deform in response to an applied
    electric charge. Applying an alternating electrical signal to the
    piezoelectric elements causes them to flex back and forth, generating a
    mechanical wave.

    Matching layers are placed between the piezoelectric elements and the
    tissues to be imaged in order to improve the energy transmission into the
    body. The acoustic impedances in the piezoelectric elements and in the body
    of the patient are poorly matched, by using one or several matching layers
    we can gradually match the impedances to limit unwanted reflections and make
    sure the energy ends up in the right place.

    An acoustic lens, much like a traditional lens, focuses the beam as it
    moves into the patient. This is also the outermost layer of the transducer,
    meaning it forms the contact surface between the transducer and the
    patient's skin.
14. Using an ultrasound transducer array lets you take advantage of the
    interference between the different transducer node elements to shape and
    steer the output signal. Depending on the complexity of the array you could
    form anything from a simple unfocused beam to complicated 3D shapes.

    The most simple transducer array is a *linear array*, meaning there are
    multiple transducers in a straight line. More complex array shapes include
    curved arrays (elements placed along a convex/concave surface) or even a
    full two-dimensional grid.

    We often talk about *phased arrays*, which is a type of transducer array
    where you have a bunch of individually controllable elements packed tightly
    together. It's called "phased" because we use *phase delays* (precise delays
    smaller than the period of our signal) in order to gain a lot of control
    over the direction and shape of our pulse.
15. There are many aspects to consider when designing an ultrasound transducer
    and depending on the specific usecase one might prioritize different things.
    In general however, the main (or most basic/essential) desired properties of
    and ultrasound transducer are:

    Being able to transmit short pulses. Shorter pulses means higher bandwidth
    means better resolution.

    Being able to transfer energy into tissues efficiently. Good acoustic
    impedance matching between the transducer and the patient means less energy
    is wasted, meaning we can use less powerful pulses. This also means there
    won't be all sorts of stray reflections/echoes all over the place muddying
    up the signal. Finally, impedance matching helps improve depth penetration.

16. The transducer bandwidth describes the range of frequencies emitted by the
    transducer. This figure is often expressed as a fraction of the center
    frequency, this is known as fractional bandwidth. The main reason we care
    about the bandwidth is because wider wavelength is linked to shorter pulses,
    which in turn means you have a smaller spatial pulse length and thus a
    better axial/radial resolution.

    Another reason we care about the bandwidth is that different tissues respond
    differently to different frequencies. So if we have more frequencies present
    in our signal (wider bandwidth), that also means we have more information
    available to better differentiate tissues.
17. The ultrasound transducer is a physical object that comprises a resonant
    electro-mechanical system. Having to exist in the real world means its
    properties are limited by the materials and manufacturing methods available.
    Todays high-end transducers use piezocermaics and are able to achieve
    fractional bandwidths up to `~80%-90%`.
18. "Backing" is the material placed behind the piezo electric element(s) in
    order to dampen any vibrations that would otherwise move backwards and
    potentially cause noise/interference.
19. Receive filtering is when you filter the echoes received back from the
    tissues to clean up and shape the signal before further processing. The
    receive filtering step usually consists of a bandpass filter to remove any
    parts of the received signal outside the region of interest (typically a
    region surrounding the center frequency of the transmitted pulse) and
    improve the signal-to-noise ratio.
20. IQ-demodulation is a way to extract amplitude and phase information from
    the received ultrasound signal by convolving the received signal with a
    reference oscillator at the center frequency. This results in two output
    signals, `I` and `Q`, which are in-phase and 90° out of phase (Quadrature),
    respectively.

    The extracted signals together contain the same information as the original,
    but split in two and shifted to a lower baseband frequency to make them
    easier to process.

## Ultrasound image formation

21. Diffraction is the bending/spreading of a wave as it encounters an obstacle
    or passes through a small opening. This happens in small tissue interfaces
    or vessel openings and also at the edges of the ultrasound beam.
    Diffraction causes the beam to spread out, especially at low frequencies.
    This reduces lateral resolution as the beam spreads and creates blur around
    small structures. We use beamforming techniques like focusing to help
    reduce/control the effects of diffraction.

    Refraction is a change in direction caused by passing between different
    mediums with different speeds of sound. Refraction happens when the
    ultrasound wave passes through the interface between different materials
    (such as where fat meets muscle) at an oblique angle. This can cause lateral
    displacement in the image and/or produce unwanted image artefacts like
    duplicate structures caused by unwanted echoes.
22. A beamprofile is a way to characterize the shape and behavior of an
    ultrasound wave. The unfocused beamprofile specifically, means the
    beamprofile when no electronic or mechanical focusing/shaping is applied.
    Since we are not using anything to alter the signal, the beamprofile is
    entirely dependant on the physical shape/layout of the transducer.

    The overall shape of the unfocused beam is that of an hourglass, but because
    of the backing material of the transducer we typically only deal with one
    half of the hourglass as the other half is dampened.

    The unfocused beamprofile has two zones to consider: the near field (fresnel
    zone) and the far field (fraunhofer zone). In the near field we see both
    constructive and destructive interference patterns and an overall
    non-uniform beam (width varies with distance). In the far field, the wave
    will have taken on a more uniform shape and the beam diverges steadily with
    distance.
23. The Frauenhofer approximation states that a wave emitted from an aperture,
    when far enough away from the source (far field), will have a diffraction
    pattern proportional to the fourier transform of the aperture velocity
    distribution function. In other words, the shape of the pressure field (in
    the Frauenhofer region) will depend on the shape of the aperture. This means
    the Frauenhofer approximation can be used to estimate the beam shape in the
    far field.
24. The "field in focus" refers to the spatial region where our beam is
    narrowest with the highest intensity. This is the region with the best
    resolution. Since the fourier transform of a rectangle implies constant
    velocity, the math governing the beamforming becomes a lot simpler and both
    fresnel (near field) and fraunhofer (far field) approximations are very
    accurate representations.
25. Beamforming is the process of manipulating a wave in order to control the
    shape of the beam formed by the pulse. Transmit-beamforming is the
    beamforming which takes place *before* the pulse is sent into the patient
    and is used to focus the transmitted energy to a specific region in the
    tissue. This is done by weigthing and delaying pulses from individual
    transducer elements. The two main forms of transmit-beamforming are steering
    (linear delay. Imposes an angle on output beam) and focusing (parabolic
    delay. Used to "squeeze" more resolution into the relevant area). It is
    possible to apply varying levels of steering and focusing at the same time.

    Receive-beamforming on the other hand is when you use beamforming
    to combine incoming echoes and shape the signal in order to listen in a
    specific direction or focus on a specific region. Like with
    transmit-beamforming, receive-beamforming also consists of weighting and
    delaying signals for individual elements, but here we are weighting/delaying
    the incoming pulses. It is possible to update/adjust the weigths and delays
    "on the fly", you could even adjust both for each new time sample.
26. Apodization is a beamforming technique where you concentrate more energy
    into the main lobe of your pulse while suppressing side lobes. This is
    achieved by exciting elements closer to the center with a greater amplitude
    than those closer to the edges, essentially having the amplitude taper off
    as you go further from the center. Apodization decreases the size of the
    side-lobes which can help improve contrast and limit image artefacts.
    However, moving the energy from the side-lobes to the main-lobe usually
    also makes the main-lobe wider, which has a negative impact on lateral
    resolution.
27. Dynamic focusing is a technique where you adapt your receive delays so
    that each point along the depth-axis is in focus. Essentially, instead of
    having one fixed focus for your image, you adapt the system to continuosly
    change the focus as new echoes come in. Kind of like moving your eyes to
    different words/lines in a book as opposed to staring at the whole page.
28. Much like dynamic focusing, using a dynamic/expanding aperture is a way to
    achieve better or mroe uniform resolution. The way this is achieved is by
    altering the size of the aperture (expanding it as the depth increases).

    Lateral resolution is described by the F-number, which is the ratio between
    the depth and the aperture size. By changing the aperture size for different
    depths, we can keep the F-number constant and have a uniform resolution.
29. A-mode, or Amplitude-mode shows intensity (amplitude) as a function of
    depth. The resulting image is a 1D slit where peaks correspond to reflectors
    and can be used to discern things like the boundaries of organs. A-mode is
    the simplest form of ultrasound imaging, and also the earliest one devised.
    It is not used much in modern medicine, except for precise measurements of
    tissue layers and certain eye exams.

    B-mode, or Brightness mode, is the most common form of ultrasound imaging in
    use today. It creates a 2D greyscale image by emitting/receiving a bunch of
    pulses in different slices (usually achieved by electronically steering the
    beam to different regions in quick succession) and combining them into one
    image. The output image axes are lateral (x) and depth (z) with the
    brightness of each pixel corresponding to the received echo strength.

    M-mode, or Motion-mode, captures motion over time along a fixed line. The
    output is an image whose axes are time and depth. This imaging mode
    works by repeatedly sending ultrasound pulses in the same direction in rapid
    succession.
30. To generate the B-mode image, we need to do a few different things:
    send/receive ultrasound waves in slices, apply filtering (TGC, bandpass),
    detect the amplitude envelope, compress/normalize intensities and finally
    combine the slices into a full image.

    Firstly, we use beamforming techniques to steer our wave/pulse to different
    regions/slices. Then we need to filter the signal, a simple bandpass filter
    gets rid of unwanted frequency components and general noise, but we also
    want to account for the attenuation due to depth. This is where TGC, or
    Time-Gain-Compensation comes in (essentially we apply the inverse function,
    amplifying depending on depth to get back to a depth-independent signal).
    Detecting the envelope can be done by taking the absolute value of the
    Hilbert transform of our signal (another way is using IQ-demodulation).
    This gives us a smooth amplitude curve we can use to estimate the signal
    strength for each received pulse, for example using FWHM. Since we are
    limited in the number of different intensities a screen can show and a
    human eye can discern, we usually also want to compress and normalize the
    intensities to a sensible range so the most relevant parts of the image are
    visible and have good contrast. Do all of that a bunch of times in a row to
    get different slices, then put them next to each other to display the
    completed B-mode image!
