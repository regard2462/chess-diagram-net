// Reads a picture file for the check and training scripts: PNG with png-lite, other formats (JPEG, WebP …) with sharp,
// which comes with the dependencies. readImage(file) → { width, height, data: RGBA bytes }.
const fs = require('fs');
const png = require('./png-lite');

async function readImage(file) {
  if (/\.png$/i.test(file)) return png.decode(fs.readFileSync(file));
  const sharp = require('sharp');
  const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data };
}

module.exports = { readImage };
