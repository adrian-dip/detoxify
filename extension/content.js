
(function () {
    chrome.storage.local.get(['enabled', 'deactivateUrls', 'severity'], data => {
        if (data.enabled) {
            var urlcheck = data.deactivateUrls
            const urlval = !urlcheck.includes(document.location.origin.split('/')[2])
            if (urlval) {
            var stopwords = "{\"test\": 0,\"auto\": 0,\"leftbox\": 0,\"float\": 0,\"left\": 0,\"width\": 0,\"height\": 0,\"px\": 0,\"font\": 0,\"size\": 0,\"px\": 0,\"box\": 0,\"sizing\": 0,\"border\": 0,\"box\": 0,\"align\": 0,\"items\": 0,\"center\": 0,\"top\": 0,\"left\": 0,\"rightbox\": 0,\"float\": 0,\"width\": 0,\"height\": 0,\"px\": 0,\"box\": 0,\"sizing\": 0,\"border\": 0,\"box\": 0,\"align\": 0,\"items\": 0,\"boxesab\": 0,\"box\": 0,\"sizing\": 0,\"border\": 0,\"box\": 0,\"explanation\": 0,\"font\": 0,\"size\": 0,\"px\": 0,\"font\": 0,\"weight\": 0,\"window\": 0,\"rin\": 0,\"background\": 0,\"rgba\": 0,\"background\": 0,\"linear\": 0,\"gradient\": 0,\"deg\": 0,\"rgba\": 0,\"rgba\": 0,\"rgb\": 0,\"height\": 0,\"px\": 0,\"width\": 0,\"px\": 0,\"color\": 0,\"border\": 0,\"radius\": 0,\"padding\": 0,\"px\": 0,\"transition\": 0,\"opacity\": 0,\"ease\": 0,\"opacity\": 0,\"position\": 0,\"fixed\": 0,\"header\": 0,\"rocket\": 0,\"height\": 0,\"px\": 0,\"main\": 0,\"rocket\": 0,\"align\": 0,\"items\": 0,\"center\": 0,\"margin\": 0,\"a\": 0, \"k\": 0,\"i'm\": 0,\"follow\": 0,\"like\": 0,\"likes\": 0,\"upvote\": 0,\"upvoted\": 0,\"yourself\": 0, \"could\": 0, \"once\": 0, \"she\": 0, \"what\": 0, \"know\": 0, \"down\": 0, \"did\": 0, \"while\": 0, \"further\": 0, \"whom\": 0, \"and\": 0, \"above\": 0, \"isn't\": 0, \"us\": 0, \"ain\": 0, \"wouldn't\": 0, \"of\": 0, \"re\": 0, \"to\": 0, \"world\": 0, \"up\": 0, \"wasn't\": 0, \"that'll\": 0, \"into\": 0, \"having\": 0, \"when\": 0, \"just\": 0, \"shouldn't\": 0, \"their\": 0, \"won\": 0, \"government\": 0, \"time\": 0, \"go\": 0, \"some\": 0, \"ourselves\": 0, \"few\": 0, \"y\": 0, \"after\": 0, \"wasn\": 0, \"am\": 0, \"going\": 0, \"now\": 0, \"below\": 0, \"by\": 0, \"where\": 0, \"you\": 0, \"you'll\": 0, \"there\": 0, \"it\": 0, \"which\": 0, \"s\": 0, \"never\": 0, \"aren\": 0, \"d\": 0, \"many\": 0, \"state\": 0, \"couldn't\": 0, \"itself\": 0, \"isn\": 0, \"money\": 0, \"were\": 0, \"mightn't\": 0, \"here\": 0, \"other\": 0, \"hers\": 0, \"the\": 0, \"mightn\": 0, \"is\": 0, \"those\": 0, \"not\": 0, \"during\": 0, \"how\": 0, \"with\": 0, \"white\": 0, \"back\": 0, \"ours\": 0, \"hasn't\": 0, \"her\": 0, \"better\": 0, \"say\": 0, \"weren\": 0, \"them\": 0, \"m\": 0, \"also\": 0, \"same\": 0, \"nothing\": 0, \"over\": 0, \"ve\": 0, \"own\": 0, \"until\": 0, \"you're\": 0, \"my\": 0, \"hadn\": 0, \"against\": 0, \"him\": 0, \"than\": 0, \"have\": 0, \"or\": 0, \"haven't\": 0, \"yours\": 0, \"these\": 0, \"work\": 0, \"but\": 0, \"again\": 0, \"you've\": 0, \"much\": 0, \"nor\": 0, \"let\": 0, \"then\": 0, \"any\": 0, \"its\": 0, \"hadn't\": 0, \"himself\": 0, \"like\": 0, \"couldn\": 0, \"tax\": 0, \"aren't\": 0, \"one\": 0, \"out\": 0, \"that\": 0, \"trump\": 0, \"each\": 0, \"ll\": 0, \"needn't\": 0, \"under\": 0, \"all\": 0, \"this\": 0, \"don\": 0, \"both\": 0, \"at\": 0, \"do\": 0, \"really\": 0, \"take\": 0, \"doesn't\": 0, \"every\": 0, \"can\": 0, \"has\": 0, \"see\": 0, \"think\": 0, \"want\": 0, \"who\": 0, \"being\": 0, \"make\": 0, \"if\": 0, \"from\": 0, \"they\": 0, \"such\": 0, \"our\": 0, \"why\": 0, \"me\": 0, \"very\": 0, \"year\": 0, \"only\": 0, \"well\": 0, \"won't\": 0, \"we\": 0, \"would\": 0, \"been\": 0, \"through\": 0, \"mustn't\": 0, \"does\": 0, \"because\": 0, \"new\": 0, \"should've\": 0, \"shouldn\": 0, \"be\": 0, \"myself\": 0, \"first\": 0, \"herself\": 0, \"doing\": 0, \"should\": 0, \"right\": 0, \"needn\": 0, \"he\": 0, \"way\": 0, \"mustn\": 0, \"shan't\": 0, \"so\": 0, \"haven\": 0, \"had\": 0, \"i\": 0, \"didn't\": 0, \"a\": 0, \"about\": 0, \"ma\": 0, \"themselves\": 0, \"public\": 0, \"don't\": 0, \"may\": 0, \"as\": 0, \"most\": 0, \"shan\": 0, \"more\": 0, \"yourselves\": 0, \"wouldn\": 0, \"doesn\": 0, \"theirs\": 0, \"weren't\": 0, \"didn\": 0, \"was\": 0, \"country\": 0, \"president\": 0, \"years\": 0, \"another\": 0, \"said\": 0, \"pay\": 0, \"t\": 0, \"between\": 0, \"canada\": 0, \"on\": 0, \"get\": 0, \"good\": 0, \"for\": 0, \"o\": 0, \"hasn\": 0, \"in\": 0, \"his\": 0, \"are\": 0, \"will\": 0, \"it's\": 0, \"even\": 0, \"still\": 0, \"you'd\": 0, \"too\": 0, \"off\": 0, \"need\": 0, \"she's\": 0, \"no\": 0, \"people\": 0, \"your\": 0, \"an\": 0, \"before\": 0} ";
            var stopwords_dict = JSON.parse(stopwords)
            var severityScore = data.severity
            let dtxCache = {}

            function isHidden(el) {
                var style = window.getComputedStyle(el);
                return ((style.display === 'none') || (style.visibility === 'hidden'))
            }


            function parseDom() {
                var domDict = {'Ej7C2RIZ6VTdJt11294IbY7BZ45i2t':severityScore}
                var allTags = document.body.getElementsByTagName('*')
                for (var i = 0, max = allTags.length; i < max; i++) {

                    if (isHidden(allTags[i])){
                        continue
                    }

                    else {
                        if (allTags[i].innerText) {
                            var intext_burner = allTags[i].innerText.toLowerCase()
                            intext_burner = intext_burner.replace(/\n/g, " ")
                            intext_burner = intext_burner.replace(/[^a-zA-Z]+/g, '')
                            intext_burner = intext_burner.split(' ')
                            var intext = intext_burner.filter(val => !(val in stopwords_dict))
                            intext = intext.filter(val => (val.length > 1))

                            if (intext.length > 0) {
                                intext = intext.join(" ")
                                domDict[intext] = i
                    }}}
                }
                return domDict
            }



            function findAndReplace (ans) {
                n_filtered = 0
                var allTags = document.body.getElementsByTagName('*')
                for (var i = 0, max = allTags.length; i < max; i++) {

                    if (isHidden(allTags[i])){
                        continue
                    }

                    else {
                        if (allTags[i].innerText) {
                            var intext_burner = allTags[i].innerText.toLowerCase()
                            intext_burner = intext_burner.replace(/\n/g, " ")
                            intext_burner = intext_burner.replace(/[^a-zA-Z]+/g, '')
                            intext_burner = intext_burner.split(' ')
                            var intext = intext_burner.filter(val => !(val in stopwords_dict))
                            intext = intext.filter(val => (val.length > 1))
                            intext = intext.join(" ")
                            if (intext.length > 0) {
                                if (ans[intext] === 1) {
                                    allTags[i].innerText = " "
                                    n_filtered++
                                    dtxCache[intext] = ans[intext]
                                } else {
                                    dtxCache[intext] = ans[intext]
                                }
                            } 
                
                chrome.storage.local.get('total', data => {
                    var total = data.total
                    var newTotal = total + n_filtered
                    chrome.storage.local.set({total:newTotal})
                })
                   
            }}}}    
            
            function findAndReplaceM(ans) {
                n_filtered = 0
                for (const res of Object.keys(ans)) {
                    if (ans[res] === 1) {
                        var domElement = mutref[res]
                        domElement.innerText = " "
                        n_filtered++
                        dtxCache[res] = ans[res]
                    } else {
                        dtxCache[res] = ans[res]
                    }
                }
                
                chrome.storage.local.get('total', data => {
                    var total = data.total
                    var newTotal = total + n_filtered
                    chrome.storage.local.set({total:newTotal})   
                })} 
            
            let mutref = {}
            let mutarr = []
            let mutationdict = {'Ej7C2RIZ6VTdJt11294IbY7BZ45i2t':severityScore}

            let observer = new MutationObserver(mutations => {  
                mutarr.push(mutations)
                if (mutarr.length > 5) {
                    var newnodes = 0
                    for (var i = 0; i < mutarr.length; i++) {  
                        for(const mutation of mutarr[i]) {
                            if (mutation.type === 'childList') {
                                for (const nnm of mutation.addedNodes) {
                                            if (nnm.innerText) {
                                                nnmNodes = nnm.getElementsByTagName('*')
                                                for (let i = 0; i < nnmNodes.length; i++) {
                                                    
                                                    if (isHidden(nnmNodes[i])) {
                                                        continue
                                                    }
                                                    else { 
                                                        if (nnmNodes[i].innerText) {
                                                        var intext_burner = nnmNodes[i].innerText.toLowerCase()
                                                        intext_burner = intext_burner.replace(/\n/g, " ")
                                                        intext_burner = intext_burner.replace(/[^a-zA-Z]+/g, '')
                                                        intext_burner = intext_burner.split(' ')
                                                        var intext = intext_burner.filter(val => !(val in stopwords_dict))
                                                        intext = intext.filter(val => (val.length > 1))
                                                        if (intext.length > 0) {
                                                            intext = intext.join(" ")
                                                            if (intext in dtxCache) {
                                                                if (dtxCache[intext] === 1) {
                                                                    nnmNodes[i].innerText = " "
                                                                }
                                                                else {
                                                                    continue
                                                                }
                                                            } 
                                                            
                                                            else {

                                                                if (!(intext in mutationdict)) {
                                                                    mutationdict[intext] = 0 
                                                                    mutref[intext] = nnmNodes[i]
                                                                    newnodes++
                                                                }

                                                                if (newnodes > 20) {
                                                                    jm =  JSON.stringify(mutationdict)
                                                                    var msgm = {
                                                                        msgtype: "contentdom",
                                                                        order: 'mutations',
                                                                        payload: jm,
                                                                    }
                                                                    mutationdict = {'Ej7C2RIZ6VTdJt11294IbY7BZ45i2t':severityScore}
                                                                    mutarr = []
                                                                    chrome.runtime.sendMessage(msgm, findAndReplaceM)
                                                                    newnodes = 0
                                                                }
                                                                else {
                                                                    continue
                                                                }
                                                            }}}}}}}}}}}
                    if (newnodes > 0) {
                        jm =  JSON.stringify(mutationdict)
                        var msgm = {
                            msgtype: "contentdom",
                            order: 'mutations',
                            payload: jm,
                        }
                        mutationdict = {'Ej7C2RIZ6VTdJt11294IbY7BZ45i2t':severityScore}
                        mutarr = []
                        chrome.runtime.sendMessage(msgm, findAndReplaceM)
                        newnodes = 0
                }})


            ///////      Messagging     ///////

            observer.observe(document, { childList: true, subtree: true })
            chrome.runtime.onMessage.addListener(gotMessage)
        
            function gotMessage(message, sender, sendResponse) {
                if (message.order === 'startup') {
                    var allDocElements = parseDom()
                    var jdom = JSON.stringify(allDocElements)
                    sendResponse(jdom)
                }

                else if (message.order ==='replace') {
                    findAndReplace(message.payload)
                }
                else if (message.order ==='replaceM'){
                    findAndReplaceM(message.payload)
                }
            }
}}})})();
