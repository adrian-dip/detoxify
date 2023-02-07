// Startup

(function () {
    chrome.tabs.query({active: true, lastFocusedWindow: true}, tabs => {
    let url = tabs[0].url
    let { hostname } = new URL(url)
    document.getElementById("website").innerHTML = hostname
    gHostname = hostname

    chrome.storage.local.get(['enabled', 'severity', 'deactivateUrls']
    ).then(function (r) {
        if (typeof r.enabled === 'undefined') {
            chrome.storage.local.set({enabled: true})
            document.getElementById("masterSwitch").checked = true
        }
        if (r.enabled === false) {
            document.getElementById("masterSwitch").checked = false
        }
        else {
            document.getElementById("masterSwitch").checked = true
        }

        if (typeof r.severity === 'undefined') {
            chrome.storage.local.set({severity: 3})
            document.getElementById("range").value = 3
        }
        else {
            document.getElementById("range").value = r.severity
        }

        if (typeof r.deactivateUrls === "undefined") {
            var ddeactivateUrls = {0:0}
            chrome.storage.local.set({deactivateUrls: ddeactivateUrls})
        }
        
        else {
            var urls = r.deactivateUrls
            if (hostname in urls) {
                document.getElementById('urlSwitch').checked = true
            }
            else {
                document.getElementById('urlSwitch').checked = false
            }
        }
    })

})})();


var enabled = true
var masterSwitch = document.getElementById('masterSwitch');
var urlSwitch = document.getElementById('urlSwitch');
var severityRange = document.getElementById('range');
var total_filtered = document.getElementById('total_filtered');


chrome.storage.local.get('enabled', data => {
    enabled = !!data.enabled
});

urlSwitch.onchange = () => {
    chrome.storage.local.get('deactivateUrls', data => {
        var urls = data.deactivateUrls
        if (gHostname in urls) {
            delete urls[gHostname] 
            chrome.storage.local.set({deactivateUrls: urls})
        }

        else {
        urls[gHostname] = 0
        chrome.storage.local.set({deactivateUrls: urls})
    }})
};

severityRange.onchange = () => {
    var severityValue = severityRange.value
    chrome.storage.local.set({severity: severityValue})
}

masterSwitch.onchange = () => {
    enabled = !enabled
    chrome.storage.local.set({enabled:enabled})
};

chrome.storage.local.get('total', data => {
    if (typeof data.total === 'undefined') {
        var total = 0
        chrome.storage.local.set({total:total})
        total_filtered.innerText = total
    }
    else {
        var total = data.total
        total_filtered.innerText = total
    }})


