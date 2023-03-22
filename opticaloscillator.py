"""
9_4_2021

Sources: (Oscillator formulation taken from Transmission Line Theory)
[1] R. Ulrich (Mar 1967). "Far infrared properties of metallic mesh and its complementary structure". Infrared Physics. 7 (1): 37–50.
[2] Sternberg, Oren. Resonances of periodic metal-dielectric structures at the infrared wavelength region. New Jersey Institute of Technology, 2002
"""
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Oscillator:
    """
    This class will calculate a set of N+2 inductive meshes.
    """

    def __init__(self, a1, A1, g, l0):
        self.g = g  # periodicity metric units
        self.w = np.linspace(0.005, 0.1, 159)
        self.l0 = l0
        self.w23 = self.g / self.l0
        self.a1 = a1
        self.A1 = A1
        self.config = True  # True normal oscillator, False Inverted oscillator
        self.n1 = 1.0  # refractive index n1
        self.n2 = 1.0  # refractive index n2
        self.n3 = 1.0  # refractive index n2
        self.omega23 = (self.w * self.g / self.w23) - (self.w23 / (self.g * self.w))  # generalized frequency

    def matrixElementsM3(self):
        """
        Matrix for mesh from index n1 to n2 M12
        """
        if self.config:
            self.Y23 = 1 / (self.a1 - (1j * self.w23 * self.A1) / (self.omega23))  # normalized admittance
        else:
            self.Y23 = 1 / (self.a1 + 1 / ((1j * self.w23 * self.A1) / (self.omega23)))

        M311 = -(self.Y23 / (2 * self.n2)) + (self.n2 + self.n3) / (2 * self.n2)
        M321 = (self.Y23 / (2 * self.n2)) + (self.n2 - self.n3) / (2 * self.n2)
        M312 = -(self.Y23 / (2 * self.n2)) + (self.n2 - self.n3) / (2 * self.n2)
        M322 = (self.Y23 / (2 * self.n2)) + (self.n2 + self.n3) / (2 * self.n2)

        self.M3 = np.array([[M311, M312], [M321, M322]], dtype=complex, order='F')

    def OscillatorPhaseGroupDelayDispersion(self):
        self.phi = (180 / np.pi) * np.arctan(np.imag(self.tf1) / np.real(self.tf1))
        self.group_delay = np.gradient(self.phi, self.w)
        self.dispersion = np.gradient(self.group_delay, self.w)

    def filteroutput(self):
        print('A1 -> {}'.format(self.A1))
        self.matrixElementsM3()
        self.tf1 = 1 / self.M3[1, 1]
        self.absTF1 = np.abs(self.tf1)
        self.TF1 = self.tf1 * np.conjugate(self.tf1)


class MixingOscillators(object):
    """
    Class to take two oscillators in different planes (treat as uncoupled)
    and mix amplitude values.
    """

    def __init__(self, xTF1, yTF1):
        self.xTF1 = xTF1
        self.yTF1 = yTF1
        self.samples = np.linspace(0.005, 0.1, 159)
        self.Z = []

    def mix(self):
        """
        Create a grid to map on 2-D and apply basic averaging of oscillators
        with no interaction assumptions.
        """
        self.xAxis = self.samples
        self.yAxis = self.samples

        self.X, self.Y = np.meshgrid(self.xAxis, self.yAxis)

        for x in range(0, len(self.xAxis), 1):
            for y in range(0, len(self.yAxis), 1):
                self.Z.append(np.sqrt(self.xTF1[x] ** 2 + self.yTF1[y] ** 2))
        self.Z = np.array(self.Z).reshape(len(self.xAxis), len(self.xAxis))

    def plotmix(self, n):
        """
        Plot the mixed oscillator output.
        """
        plot2 = plt.figure(n)
        fig, ax = plt.subplots()
        origin = 'lower'
        CS = ax.contourf(self.xAxis, self.xAxis, self.Z, 100, cmap=plt.cm.jet, origin=origin)
        ax.set_title('Uncoupled Dual Optical Oscillators')

    def plotmix_plotly(self):
        """
        Plot the mixed oscillator output using plotly.
        """
        fig = go.Figure(data=
                        go.Contour(z=self.Z, x=self.xAxis, y=self.yAxis,
                                   contours=dict(start=0.1, end=1.2, size=0.01),
                                   contours_coloring='heatmap', line_smoothing=0.85, colorscale='jet'))
        fig.write_html("./uncoupled_oscillator.html", auto_open=True)


if __name__ == "__main__":
    # Example of a single oscillator plotting absolute, real and complex components
    meshConfig1 = Oscillator(a1=0.001, A1=0.01, g=24, l0=32)
    meshConfig1.config = True
    meshConfig1.filteroutput()
    meshConfig1.OscillatorPhaseGroupDelayDispersion()
    plot1 = plt.figure(1)
    plt.plot(meshConfig1.w, meshConfig1.absTF1, label='ABS TF1')
    plt.plot(meshConfig1.w, meshConfig1.TF1, label='TF1')
    plt.plot(meshConfig1.w, np.real(meshConfig1.tf1), 'bx', label='Re[tf1]')
    plt.plot(meshConfig1.w, np.imag(meshConfig1.tf1), label='IM[tf1]')
    plt.legend(loc="upper right")
    plt.grid('True')
    plt.xlabel(r'$\omega$')

    # Plotting phase
    plot2 = plt.figure(2)
    plt.plot(meshConfig1.w, meshConfig1.phi, label='Phase')
    plt.legend(loc="upper right")
    plt.xlim([0, 0.1])
    plt.ylim([-100, 100])

    plot3 = plt.figure(3)
    plt.plot(meshConfig1.w, meshConfig1.group_delay, label='Group Delay')
    plt.legend(loc="upper right")

    # Plotting dispersion relation
    plot4 = plt.figure(4)
    plt.plot(meshConfig1.w, meshConfig1.dispersion, label='Dispersion Relation')
    plt.legend(loc="upper right")

    # Second oscillator
    meshConfig2 = Oscillator(a1=0.01, A1=3, g=10, l0=15)
    meshConfig1.config = False
    meshConfig2.filteroutput()
    meshConfig2.OscillatorPhaseGroupDelayDispersion()

    # Mixing oscillators with no coupling
    mix_uncoupled_oscillators = MixingOscillators(meshConfig1.absTF1, meshConfig2.absTF1)
    mix_uncoupled_oscillators.mix()
    mix_uncoupled_oscillators.plotmix(n=10)
    plt.show()

    # Plotly/Dash plots [needs updating]
    mix_uncoupled_oscillators.plotmix_plotly()