import numpy as np
import numba

from . import find


class Finder(find.Finder):
    max_num_peaks = 10000
    absolute_max_num_peaks = 1000000

    def local_maxima(self, image, threshold):
        while (self.max_num_peaks < self.absolute_max_num_peaks):
            idx_of_max = np.empty((self.max_num_peaks, 2), dtype=np.int)

            num_peaks = _numba_local_maxima(
                idx_of_max, image, self.background + threshold, self.peak_count,
                self.search_radius, self.margin)

            if num_peaks >= 0:
                break

            self.max_num_peaks = min(self.max_num_peaks*10,
                                     self.absolute_max_num_peaks)

        idx_of_max.resize((num_peaks, 2))
        return idx_of_max


@numba.jit(nopython=True)
def _numba_local_maxima(idx_of_max, image, threshold, peak_count,
                        search_radius, margin):
    r = int(search_radius + 0.5)
    sr2 = search_radius * search_radius
    cnt = 0
    max_cnt = len(idx_of_max)

    for i in range(margin, image.shape[0] - margin + 1):
        for j in range(margin, image.shape[1] - margin + 1):
            pix_val = image[i, j]
            if pix_val <= threshold:
                continue

            is_max = True
            for k in range(-r, r+1):
                for l in range(-r, r+1):
                    if k*k + l*l >= sr2:
                        continue
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

            if cnt >= max_cnt:
                return -1

            idx_of_max[cnt, 0] = i
            idx_of_max[cnt, 1] = j
            cnt += 1

    return cnt
