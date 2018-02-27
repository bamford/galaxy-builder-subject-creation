import os
import json
import gzip
import shutil
from numpy import sum, max, array
import astropy.units as u
from astropy.io import fits
import sdssCutoutGrab as scg
import createSubjectsFunctions as csf


# TODO: replace indexing with some kind of proper object ID
def main(objList, outFolder='subjects', outFolderFits='cutouts'):
    # objList is list of (ra, dec, petrotheta)s for target galaxies
    # make sure the output folders exists
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)
    if not os.path.exists(outFolderFits):
        os.mkdir(outFolderFits)
    psfs = []
    # cycle through the input objects
    for i, (ra, dec, petrotheta) in enumerate(objList):
        # search by ra, dec
        print('🛰  Looking for galaxy at {}, {}'.format(ra, dec))
        frame = scg.queryFromRaDec(ra, dec)
        if not len(frame):
            print("💩  Couldn\'t find any galaxies")
            continue
        fileLoc = scg.getBandFits(frame)
        fitsFile = fits.open(fileLoc)
        # read it in and crop out around the galaxy
        imageData, sigma = scg.cutFits(
            fitsFile,
            ra, dec,
            size=(4 * petrotheta * u.arcsec, 4 * petrotheta * u.arcsec)
        )
        if imageData is False:
            print('\t💀  \033[31mReturned False from image Data\033[0m')
            print('\tRa: {} Dec: {}'.format(ra, dec))
            continue

        # Use source extractor to identify objects TODO proper deblending
        objects, segmentation_map = csf.sourceExtractImage(
            imageData,
            fitsFile[2].data[0][0]
        )
        # create a true/false masking array
        mask = csf.maskArr(imageData, segmentation_map, objects[-1][0] + 1)

        # create the masked image
        maskedImageData = imageData[:]
        maskedImageData[mask] = 0

        # apply an asinh stretch and save the image to the outfolder
        resizeTo = (512, 512)
        csf.saveImage(
            csf.stretchArray(maskedImageData[:, ::-1]),
            fname="{}/image_{}.png".format(outFolder, i),
            resize=True,
            size=resizeTo
        )
        # Now we find the PSF
        psf = csf.getPSF((ra, dec), frame, fitsFile)
        c = 20
        # crop out most of the 0-ish stuff
        psfCut = psf[c:-c, c:-c]
        # normalise so we don't lose flux
        psfCut = psfCut / sum(psfCut)
        psfs += [[str(i) for i in psfCut.reshape(psfCut.size)]]

        # generate the model json
        model = {
            'psf': psfCut.tolist(),
            'psfWidth': psfCut.shape[1],
            'psfHeight': psfCut.shape[0],
            'width': imageData.shape[1],
            'height': imageData.shape[0],
            'imageWidth': resizeTo[0],
            'imageHeight': int(
                imageData.shape[0] / imageData.shape[1] * resizeTo[0]
            )
        }
        # and the difference json
        difference = {
            'psf': psfCut.tolist(),
            'psfWidth': psfCut.shape[1],
            'psfHeight': psfCut.shape[0],
            'mask': array(mask, dtype=float).tolist(),
            'imageData': (imageData / max(imageData)).tolist(),
            'size': imageData.shape[1],
            'height': imageData.shape[0],
            'imageWidth': resizeTo[0],
            'imageHeight': int(
                imageData.shape[0] / imageData.shape[1] * resizeTo[0]
            )
        }
        # and the subject metadata
        metadata = {
            '#originalBrightness': float(max(imageData)),
            'Ra': ra,
            'Dec': dec,
            'petrotheta': petrotheta,
            'SDSS_ID': int(frame.get('objID', 0)),
            '#isModelling': True,
            '#models': [
                {'frame': 1, 'model': 'GALAXY_BUILDER_DIFFERENCE'},
                {'frame': 2, 'model': 'GALAXY_BUILDER_MODEL'},
            ]
        }
        # write out the model (saving a gzipped and non-gzipped version)
        modelFileName = '{}/model_{}.json'.format(outFolder, i)
        with open(modelFileName, 'w') as f:
            json.dump(model, f)
        with open(modelFileName, 'rb') as f_in, \
                gzip.open(modelFileName + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # write out the difference
        diffFileName = '{}/difference_{}.json'.format(outFolder, i)
        with open(diffFileName, 'w') as f:
            json.dump(difference, f)
        with open(diffFileName, 'rb') as f_in, \
                gzip.open(diffFileName + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        metaFileName = '{}/metadata_{}.json'.format(outFolder, i)
        with open(metaFileName, 'w') as f:
            json.dump(metadata, f)

        # create galfit input file
        feedme = "{}/galfit_{}.feedme".format(outFolderFits, i)
        galfit = {'image': "{}/image_{}.fits".format(outFolderFits, i),
                  'mask': "{}/mask_{}.fits".format(outFolderFits, i),
                  'sigma': "{}/sigma_{}.fits".format(outFolderFits, i),
                  'psf': "{}/psf_{}.fits".format(outFolderFits, i),
                  'output': "{}/galfit_{}.fits".format(outFolderFits, i),
                  'xmax': imageData.shape[0],
                  'ymax': imageData.shape[1],
                  'zeropoint': 25}
        with file(feedme, 'w') as feedme_file:
            for line in file('feedme.template'):
                feedme_file.write(line.format(galfit)+'\n')

        # save fits
        fits.writeto(galfit['image'], imageData)
        fits.writeto(galfit['mask'], mask)
        fits.writeto(galfit['sigma'], sigma)
        fits.writeto(galfit['psf'], psf)


lucyGals = (
    (160.65883, 23.95189, 22.3782),
    (119.06931414139731, 11.662177891345076, 25.510479),
    (236.14108, 10.29315, 27.98376),
    (216.53496423791432, 5.237946739507734, 27.81166),
    (248.7370584891894, 25.69259397115592, 26.418024),
    (239.5076772567969, 14.963535027843466, 26.63623),
    (178.44322153341767, 10.40313979646676, 18.855566),
)

if __name__ == '__main__':
    main(lucyGals)
