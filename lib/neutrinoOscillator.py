import numpy as np
from matplotlib import pyplot as plt
import uproot

class neutrinoOscillator(object):

    def __init__(self):
        self.length = 477

    def setNumuFluxRoot(self, flux_file):
        root_file = uproot.open(flux_file)
        self.bin_edges = root_file['numu'].numpy()[1]
        self.bin_widths = (self.bin_edges[1:] - self.bin_edges[:-1])
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:])/2
        self.numu_flux = root_file['numu'].numpy()[0]
        self.nue_flux = root_file['nue'].numpy()[0]

    def setDetectorDistance(self, length):
        self.length = length

    @staticmethod
    def oscillationProbability(energy, length, deltam2=1.32, sin2theta2=0.001):
        return sin2theta2 * np.sin(1.27 * deltam2 * length / (energy*1000))**2

    def oscillationWeightAtEnergy(self, energy, deltam2=1.32, sin2theta2=0.001):
        bin_number = np.digitize(energy, self.bin_edges) - 1
        p_oscillation = self.oscillationProbability(energy, self.length, deltam2, sin2theta2)
        return p_oscillation * self.numu_flux[bin_number] / self.nue_flux[bin_number]

    @staticmethod
    def addRange(deltam2, sin2theta2):
        return (r'$\Delta m^2_{14}$ = ' +
                  '{:.3g}'.format(deltam2) +
                  r' $eV^2, \sin^2(2\theta_{e\mu})$ = ' +
                  '{:.3g}'.format(sin2theta2))

    def plotOscillationWeights(self, deltam2=1.32, sin2theta2=0.001):
        p_oscillation = self.oscillationProbability(self.bin_centers, self.length, deltam2, sin2theta2)
        plt.step(self.bin_edges[:-1], p_oscillation*100, label='oscillation probability x 100')
        osc_weights = self.oscillationWeightAtEnergy(self.bin_centers, deltam2, sin2theta2)
        plt.step(self.bin_edges[:-1], osc_weights, label=r'$\nu_e$ weight')
        plt.xlabel('Neutrino Energy [GeV]')
        plt.ylabel('Oscillation weight')
        plt.legend()
        plt.title(r'Oscillation probability and weights for $\nu_e$ events' + '\n' + neutrinoOscillator.addRange(deltam2, sin2theta2))

    def plotOscillatedSpectrum(self, deltam2=1.32, sin2theta2=0.001):
        nue_osc_flux = self.numu_flux*self.oscillationProbability(self.bin_centers, self.length, deltam2, sin2theta2)
        p1 = plt.bar(self.bin_centers, self.nue_flux, self.bin_widths, label=r'BNB $\nu_e$ flux')
        p2 = plt.bar(self.bin_centers, nue_osc_flux, self.bin_widths, bottom=self.nue_flux, label=r'BNB $\nu_{\mu} \rightarrow \nu_e$ flux')
        plt.xlabel('Neutrino Energy [GeV]')
        plt.ylabel('Neutrino Flux')
        plt.legend()
        plt.title('Electron neutrino flux at MicroBooNE\n' + neutrinoOscillator.addRange(deltam2, sin2theta2))

    def addEventRateFile(self, tfile, pot_scaling, weightToBeApplied='weightSplineTimesTune'):
        self.array = uproot.open(tfile)['nuselection']['NeutrinoSelectionFilter'].arrays(branches=['nu_e', weightToBeApplied], namedecode='utf-8')
        self.array['weightSplineTimesTune'][self.array['weightSplineTimesTune'] <= 0] = 1.
        self.array['weightSplineTimesTune'][np.isinf(self.array['weightSplineTimesTune'])] = 1.
        self.array['weightSplineTimesTune'][self.array['weightSplineTimesTune'] > 100] = 1.
        self.array['weightSplineTimesTune'][np.isnan(self.array['weightSplineTimesTune']) == True] = 1.

        self.pot_scaling = pot_scaling
        self.weightToBeApplied = weightToBeApplied

    def plotOscillatedRate(self, deltam2=1.32, sin2theta2=0.001):
        self.array['this_weight'] = self.pot_scaling* self.array[self.weightToBeApplied] * (1. + self.oscillationWeightAtEnergy(self.array['nu_e'], deltam2, sin2theta2))

        plt.hist(self.array['nu_e'],
                 bins=self.bin_edges[::2],
                 weights=self.array['this_weight'],
                 label=r'BNB $\nu_{\mu} \rightarrow \nu_e$ events')
        plt.hist(self.array['nu_e'],
                 bins=self.bin_edges[::2],
                 weights=self.pot_scaling* self.array[self.weightToBeApplied],
                 label=r'BNB $\nu_e$ events'
                 )
        plt.xlabel('Neutrino Energy [GeV]')
        plt.ylabel('Expected number of electron neutrino events')
        plt.legend()
        plt.title('Charged Current electron neutrino event rate at MicroBooNE\nMicroBooNE preliminary - 10.1e20 POT\n' + neutrinoOscillator.addRange(deltam2, sin2theta2), loc='left')