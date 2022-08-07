var HttpClient = function() {
  this.get = function(aUrl, aCallback) {
    var anHttpRequest = new XMLHttpRequest();
    anHttpRequest.onreadystatechange = function() {
      if (anHttpRequest.readyState == 4 && anHttpRequest.status == 200)
      aCallback(anHttpRequest.responseText);
    }

    anHttpRequest.open( "GET", aUrl, true );
    anHttpRequest.send( null );
  }
}

chrome.storage.local.get('key', function(result) {
  console.log(result);
  var data = JSON.stringify({text: [new String(result.key)]});
  var xhr = new XMLHttpRequest();

  xhr.open("POST", "http://127.0.0.1:5000/predict", true);
  xhr.setRequestHeader("Content-type", "application/json");
  xhr.onreadystatechange = function() {
    if (xhr.readyState == 4) { 
      console.log(xhr.responseText);
      const obj = JSON.parse(xhr.responseText);
      var pred;
      if (obj.prediction == 1){
        pred = 'REAL'
        for (let i = 0; i < 1000; i++) {
          alert('get off a here NOW')
          alert('this is real')
        }
        $('.detected').css({'display': 'none'});
      }
      else {
          pred = 'FAKE';

          $('.detected').css({'display': 'block'});
          
     
      }
      $('#results').replaceWith(pred);
      
  }
}


chrome.tabs.executeScript( {
  code: "window.getSelection().toString();"
}, function(selection) {


  fetch(configURL)
    .then((response) => {
      return response.json();
    })
    .then((data) => {
      console.log(data);

      var url = data.ip + ":" + data.port + "/?query=";
      var str = encodeURIComponent(parseTweet(selection[0]));
      var query = url.concat(str);

      var client = new HttpClient();
      client.get(query, function(res) {
        document.getElementById("loading").style.display = "none";
        document.getElementById("output").style.display = "inline-block";
      });

    });

});

console.log(data)
xhr.send(data);
  });

function getActualPrediction(pred, confidence) {
  if(confidence >= 0.8) {
    return pred;
  }
  if(confidence >= 0.6) {
    return "Possibly " + pred;
  }
  return "Unsure";
}