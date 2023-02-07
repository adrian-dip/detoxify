chrome.storage.local.get(['enabled', 'deactivateUrls', 'severity'], data => {
    const severityScore = data.severity

    function doStuffWithDom(domContent) {
        var options = {
            method: 'POST',
            headers: {
                "Content-Type": "application/json"
                },
            body: domContent
        }
        
        fetch('http://dipadrian.pythonanywhere.com', 
        options).then(function(r) {
            return r.json()
        }).then(function (data) {
            var answers = data;
            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                let current_tab = tabs[0]
                if (current_tab.url) {
                    let msg_a = {
                        msgtype: "contentdom",
                        order: 'replace',
                        payload: answers
                        };
                    chrome.tabs.sendMessage(current_tab.id, msg_a)
                }
        })})
    }

    function doStuffWithDomM(domContent) {
        var options = {
            method: 'POST',
            headers: {
                "Content-Type": "application/json"
                },
            body: domContent
            }
        fetch('http://dipadrian.pythonanywhere.com', 
        options).then(function(r) {
            return r.json()
        }).then(function (data) {
            var answers = data
            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                let current_tab = tabs[0]
                if (current_tab.url) {
                    let msg_a = {
                        msgtype: "contentdom",
                        order: 'replaceM',
                        payload: answers
                    };
                    chrome.tabs.sendMessage(current_tab.id, msg_a)
                }
        })})
    }



    chrome.tabs.onUpdated.addListener(function(tab) {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        let current_tab = tabs[0]
        if (current_tab.url) {

            let msg = {
                msgtype: "contentdom",
                order: 'startup'
            };
            
            chrome.tabs.sendMessage(current_tab.id, msg, doStuffWithDom);
            }
        } 
    )});

    chrome.runtime.onMessage.addListener(gotMessageB);

    function gotMessageB(message, sender, sendResponse) {
        if (message.order === 'mutations') {
            doStuffWithDomM(message.payload)
        }
    }
})