'use strict';

var masks = [
  'VanillaGrad',
  'IntegGrad',
  'GuidedBackProp',
  'NoisyGrad',
  'IntegGrad+NoisyGrad',
  'GuidedBackProp+NoisyGrad'
];

var imgxmasks = [
  'VanillaGradIm',
  'IntegGradIm',
  'GuidedBackPropIm',
  'NoisyGradIm',
  'IntegGradIm+NoisyGrad',
  'GuidedBackPropIm+NoisyGrad'
];

var mode = 0; // masks or imgxmasks

var idImgMap = {};
var idLoadedMap = {};
var idClickMap = {};
// Map of id to 0 (with respect to label) or 1 (with respect to prediction).
var idWrtMap = {};

// Array of checkpointed divs.
var checkpoints = [];
var checkpoint_freq = 10;

var tbody = document.querySelector('#table tbody');
var xhr = new XMLHttpRequest();
xhr.open('GET', 'metadata.json');
xhr.onload = () => {
  var metadata = JSON.parse(xhr.responseText);

  metadata.forEach((row, ogid) => {
    var i = row["id"] - 1;

    var topPred = row.top5[0][0].split(',')[0];
    var label = row.label.split(',')[0];

    var isCorrectPrediction = true;
    if (topPred !== label) {
      isCorrectPrediction = false;
    }

    idImgMap[i] = [];
    idLoadedMap[i] = false;
    idWrtMap[i] = 0;

    // Header
    var trheader = document.createElement('tr');
    trheader.className = 'label-row ' + (!isCorrectPrediction ? 'incorrect-row' : 'correct-row');

    var tdspacer = document.createElement('td');
    trheader.appendChild(tdspacer);

    var labelcell = document.createElement('td');
    labelcell.colSpan = 5;
    if (isCorrectPrediction) {
      labelcell.innerHTML = '<i>Label: ' + label + '</i>';
    } else {
      labelcell.innerHTML = 'Show gradient of:';

      var linksContainer = document.createElement('div');

      var labelLink = document.createElement('a');
      labelLink.innerText = 'Label: ' + label;
      labelLink.className = 'selected-gradient-wrt gradient-wrt';

      var predictionLink = document.createElement('a');
      predictionLink.innerText = 'Prediction: ' + topPred;
      predictionLink.className = 'gradient-wrt';

      linksContainer.appendChild(labelLink);
      linksContainer.appendChild(document.createElement('br'));
      linksContainer.appendChild(predictionLink);
      labelcell.appendChild(linksContainer);

      $(labelLink).click(function(clickedLink, otherLink, id) {
        $(clickedLink).addClass('selected-gradient-wrt');
        $(otherLink).removeClass('selected-gradient-wrt');
        idWrtMap[id] = 0;

        var images = idImgMap[id];
        var newattr = mode === 0 ? 'data-mask-src' : 'data-imgxmask-src';
        for (var i = 0; i < images.length; i++) {
          images[i].src = images[i].getAttribute(newattr);
        }
      }.bind(null, labelLink, predictionLink, i));

      $(predictionLink).click(function(clickedLink, otherLink, id) {
        $(clickedLink).addClass('selected-gradient-wrt');
        $(otherLink).removeClass('selected-gradient-wrt');
        idWrtMap[id] = 1;

        var images = idImgMap[id];
        var newattr = mode === 0 ? 'data-mask-wrt-pred-src' : 'data-imgxmask-wrt-pred-src';
        for (var i = 0; i < images.length; i++) {
          images[i].src = images[i].getAttribute(newattr);
        }
      }.bind(null, predictionLink, labelLink, i));

    }

    labelcell.className = 'image-label';
    trheader.appendChild(labelcell);
    if (i % checkpoint_freq == 0) {
      checkpoints.push(trheader);
    }

    tbody.appendChild(trheader);

    // Images.
    var tr = document.createElement('tr');
    tr.className = 'images-row ' + (!isCorrectPrediction ? 'incorrect-row' : 'correct-row');
    tbody.appendChild(tr);

    // ID
    var td = document.createElement('td');
    tr.appendChild(td);
    td.style.width = '4%';
    td.innerHTML = (i + 1);
    td.style.fontSize = '20px';

    // Image.
    td = document.createElement('td');
    td.style.width = '32%';
    td.className = 'main-img';
    td.style.position = 'relative';
    tr.appendChild(td);

    var img = document.createElement('img');
    img.setAttribute('data-mask-src', 'images/' + (i+1) + '.png');
    img.setAttribute('data-imgxmask-src', 'images/' + (i+1) + '.png');
    img.setAttribute('data-mask-wrt-pred-src', 'images/' + (i+1) + '.png');
    img.setAttribute('data-imgxmask-wrt-pred-src', 'images/' + (i+1) + '.png');

    img.style.position = 'absolute';
    img.style.zIndex = 10;
    img.style.top = 0;
    img.style.left = 0;
    idImgMap[i].push(img);
    img.style.width = '100%';
    img.style.height = '100%';
    img.setAttribute('data-id', i);

    var ogimg = img;
    var canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.zIndex = 15;
    canvas.style.top = 0;
    canvas.style.left = 0;
    canvas.width = 1;
    canvas.height = 1;

    td.appendChild(img);
    td.appendChild(canvas);
    tr.appendChild(td);

    // Table 2x3.
    td = document.createElement('td');
    td.className = 'grads-container';
    td.setAttribute('data-id', i);
    tr.appendChild(td);
    var table = document.createElement('table');
    table.className = 'inner';
    td.appendChild(table);

    $(canvas).click(function(i, event) {
      idClickMap[i] = false;
      canvas.style.display = 'none';
    }.bind(null, i));

    $(td).mouseout(function(i, event) {
      if (idClickMap[i]) {
        return;
      }
      canvas.style.display = 'none';
    }.bind(null, i));

    // Row 1.
    tr = document.createElement('tr');
    table.appendChild(tr);

    for (var m = 0; m < 3; m++) {
      var mask = masks[m];
      var imgxmask = imgxmasks[m];

      td = document.createElement('td');
      td.className = 'grad';
      var img = document.createElement('img');
      tr.appendChild(td).appendChild(img);
      img.setAttribute('data-mask-src', 'images/' + (i+1) + '_' + mask + '.png');
      img.setAttribute('data-imgxmask-src', 'images/' + (i+1) + '_' + imgxmask + '.png');
      if (!isCorrectPrediction) {
        img.setAttribute('data-mask-wrt-pred-src', 'images/' + (i+1) + '_' + mask + '_wrt_pred.png');
        img.setAttribute('data-imgxmask-wrt-pred-src', 'images/' + (i+1) + '_' + imgxmask + '_wrt_pred.png');
      }
      img.setAttribute('data-id', i);

      img.className = 'grad';
      idImgMap[i].push(img);

      $(img).mouseover(function(i, event) {
        if (idClickMap[i]) {
          return;
        }
        combineImageAndMask(ogimg, event.target, canvas);
      }.bind(null, i));

      $(img).click(function(i, event) {
        idClickMap[i] = true;
        combineImageAndMask(ogimg, event.target, canvas);
      }.bind(null, i));
    }

    td = document.createElement('td');
    var div = document.createElement('div');
    div.innerHTML = 'Plain';
    div.className = 'rotated';
    td.appendChild(div);
    tr.appendChild(td);

    // Row 2.
    tr = document.createElement('tr');
    table.appendChild(tr);

    for (var m = 3; m < 6; m++) {
      var mask = masks[m];
      var imgxmask = imgxmasks[m];

      td = document.createElement('td');
      td.className = 'grad';
      var img = document.createElement('img');
      tr.appendChild(td).appendChild(img);
      img.setAttribute('data-mask-src', 'images/' + (i+1) + '_' + mask + '.png');
      img.setAttribute('data-imgxmask-src', 'images/' + (i+1) + '_' + imgxmask + '.png');
      if (!isCorrectPrediction) {
        img.setAttribute('data-mask-wrt-pred-src', 'images/' + (i+1) + '_' + mask + '_wrt_pred.png');
        img.setAttribute('data-imgxmask-wrt-pred-src', 'images/' + (i+1) + '_' + imgxmask + '_wrt_pred.png');
      }

      img.setAttribute('data-id', i);
      idImgMap[i].push(img);

      $(img).mouseover(function(i, event) {
        if (idClickMap[i]) {
          return;
        }
        combineImageAndMask(ogimg, event.target, canvas);
      }.bind(null, i));
      $(img).click(function(i, event) {
        combineImageAndMask(ogimg, event.target, canvas);
        idClickMap[i] = true;
      }.bind(null, i));
    }

    td = document.createElement('td');
    div = document.createElement('div');
    div.innerHTML = 'SmoothGrad';
    div.className = 'rotated smoothgrad-label';
    td.appendChild(div);
    tr.appendChild(td);

    // Spacing footer
    var trfooter = document.createElement('tr');
    trfooter.className = 'footer-row ' + (!isCorrectPrediction ? 'incorrect-row' : 'correct-row');
    var tdfooter = document.createElement('td');
    tdfooter.colSpan = 6;
    tdfooter.style.outline = 'none';
    trfooter.appendChild(tdfooter);
    tbody.appendChild(trfooter);

  });
  readyPage();
};
xhr.send();


function getImagePixels(img, width, height) {
  var canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(img, 0, 0, width, height);
  var pixelData = canvas.getContext('2d').getImageData(0, 0, width, height).data;
  return pixelData;
}

function combineImageAndMask(img, maskImg, canvas) {
  if (img.complete && maskImg.complete) {
    canvas.style.display = 'block';
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.style.width  = img.width + 'px';
    canvas.style.height = img.height + 'px';
    var pixels = getImagePixels(img, canvas.width, canvas.height);
    var gradPixels = getImagePixels(maskImg, canvas.width, canvas.height);
    var ctx = canvas.getContext('2d');
    var imgData = ctx.createImageData(img.width, img.height);
    for (var i = 0; i < imgData.data.length; i += 4) {
      //var opacity = (gradPixels[i] / 255.0 > 0.2) ? 1 : 0.2; //Binarized mask.
      //var opacity = Math.min(1, gradPixels[i] / 150.0); // Capped mask.
      var opacity = gradPixels[i] / 255.0;
      imgData.data[i+0] = pixels[i+0] * opacity;
      imgData.data[i+1] = pixels[i+1] * opacity;
      imgData.data[i+2] = pixels[i+2] * opacity;
      imgData.data[i+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }
}

function readyPage() {
  // Make the headers fixed position.
  $(document).ready(function() {
    var headers = $('#headers');
    var tableContainer = $('#table-container');
    var headerPosition = headers.position();

    var scrollEventScheduled = false;
    $(window).scroll(function(e) {
      if (!scrollEventScheduled) {
        window.requestAnimationFrame(() => {
          updateHeader();
          scrollEventScheduled = false;
        });
      }
      scrollEventScheduled = true;
    });

    function updateHeader() {
      if (window.scrollY > headerPosition.top) {
        headers.addClass('fixed-header');
        tableContainer.addClass('fixed-header');
      } else {
        tableContainer.removeClass('fixed-header');
        headers.removeClass('fixed-header');
      }

      // check lazy loading
      for (var i = 0; i < checkpoints.length; i++) {
        if (window.scrollY > checkpoints[i].offsetTop &&
            i < (200 / checkpoint_freq) - 1) {
          loadCheckpoint(i + 1);
        }
      }
    }

    var showMispred = false;
    $("#show-mispred").change(function(event) {
      if (this.checked) {
        $("#table").addClass('show-mispred');
      } else {
        $("#table").removeClass('show-mispred');
      }
    });

    function loadCheckpoint(c) {
      for (var i = 0; i < checkpoint_freq; i++) {
        var id = checkpoint_freq * c + i;
        if (idLoadedMap[id]) {
          continue;
        }

        var images = idImgMap[id];
        for (var j = 0; j < images.length; j++) {
          var image = images[j];

          var newattr = mode == 0 ? 'data-mask' : 'data-imgxmask';
          newattr += (idWrtMap[i] === 0 ? '' : '-wrt-pred') + '-src';

          image.src = image.getAttribute(newattr);
        }
        idLoadedMap[id] = true;
      }
    }
    loadCheckpoint(0);

    function changeMode(cmode) {
      mode = cmode;

      // Only iterate through pre-loaded
      for (var i = 0; i < 200; i++) {
        if (idLoadedMap[i]) {
          var images = idImgMap[i];
          for (var j = 0; j < images.length; j++) {
            var image = images[j];

            var newattr = mode == 0 ? 'data-mask' : 'data-imgxmask';
            newattr += (idWrtMap[i] === 0 ? '' : '-wrt-pred') + '-src';
            image.src = image.getAttribute(newattr);
          }
        }
      }
    }

    updateHeader();

    var selectGrad = $('.select-grad');
    var selectImgTimesGrad = $('.select-img-times-grad');

    selectGrad.click(function(e) {
      selectGrad.addClass('grad-selected');
      selectGrad.removeClass('grad-unselected');

      selectImgTimesGrad.addClass('grad-unselected');
      selectImgTimesGrad.removeClass('grad-selected');
      changeMode(0);
    });
    $('.select-img-times-grad').click(function(e) {
      selectGrad.addClass('grad-unselected');
      selectGrad.removeClass('grad-selected');

      selectImgTimesGrad.addClass('grad-selected');
      selectImgTimesGrad.removeClass('grad-unselected');
      changeMode(1);
    });

    $(window).resize(function(event) {
      resize();
    });
    function resize() {
      $('#headers').width($('.attribution-container').width());
    }
    resize();

  });
}
