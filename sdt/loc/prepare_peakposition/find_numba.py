import numpy as np
import numba

from . import find


class Finder(find.Finder):
    max_num_peaks = 10000
    absolute_max_num_peaks = 1000000

    def local_maxima(self, image, threshold):
        while (self.max_num_peaks < self.absolute_max_num_peaks):
            idx_of_max = np.empty((self.max_num_peaks, 2), dtype=np.int)
            mass = np.empty(self.max_num_peaks)
            bg = np.empty(self.max_num_peaks)

            num_peaks = _numba_local_maxima(
                idx_of_max, mass, bg, image, threshold, self.mass_radius,
                self.bg_radius, self.search_radius,)

            if num_peaks >= 0:
                break

            self.max_num_peaks = min(self.max_num_peaks*10,
                                     self.absolute_max_num_peaks)

        idx_of_max.resize((num_peaks, 2))
        mass.resize(num_peaks)
        bg.resize(num_peaks)
        return idx_of_max, mass, bg


@numba.jit(nopython=True)
def _numba_local_maxima(idx_of_max, mass, bg, image, threshold, mass_radius,
                        bg_radius, search_radius):
    mass_area = (2*mass_radius + 1)**2
    bg_area = (2*bg_radius + 1)**2 - mass_area
    ring_size = bg_radius - mass_radius

    cnt = 0
    max_cnt = len(idx_of_max)

    for i in range(bg_radius, image.shape[0]-bg_radius):
        for j in range(bg_radius, image.shape[1]-bg_radius):

            # see whether current pixel is a local maximum
            pix_val = image[i, j]

            is_max = True
            for k in range(-search_radius, search_radius+1):
                for l in range(-search_radius, search_radius+1):
                    if (k <= 0) and (l <= 0):
                        if pix_val < image[i+k, j+l]:
                            is_max = False
                            break
                    else:
                        if pix_val <= image[i+k, j+l]:
                            is_max = False
                            break

                if not is_max:
                    break

            if not is_max:
                continue

            # calculate mass
            cur_mass = 0.
            for k in range(-mass_radius, mass_radius+1):
                for l in range(-mass_radius, mass_radius+1):
                    cur_mass += image[i+k, j+l]

            cur_bg = 0.
            # calculate background
            # upper part of the ring
            for k in range(-bg_radius, -bg_radius+ring_size):
                for l in range(-bg_radius, bg_radius+1):
                    cur_bg += image[i+k, j+l]
            # lower part
            for k in range(bg_radius-ring_size+1, bg_radius+1):
                for l in range(-bg_radius, bg_radius+1):
                    cur_bg += image[i+k, j+l]
            # left
            for k in range(-bg_radius+ring_size, bg_radius-ring_size+1):
                for l in range(-bg_radius, -bg_radius+ring_size):
                    cur_bg += image[i+k, j+l]
            # right
            for k in range(-bg_radius+ring_size, bg_radius-ring_size+1):
                for l in range(bg_radius-ring_size+1, bg_radius+1):
                    cur_bg += image[i+k, j+l]

            cur_avg_bg = cur_bg / bg_area
            cur_mass_corr = cur_mass - cur_avg_bg*mass_area

            if cur_mass_corr <= threshold:
                continue

            if cnt >= max_cnt:
                return -1

            idx_of_max[cnt, 0] = i
            idx_of_max[cnt, 1] = j
            mass[cnt] = cur_mass_corr
            bg[cnt] = cur_avg_bg
            cnt += 1

    return cnt
