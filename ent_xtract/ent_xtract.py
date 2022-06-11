from typing import Callable, Mapping, Sequence, Dict, Iterable

from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import numpy as np
import random

import spacy

class WordMan:
    def __init__(self, stemmer, lematizer):
        self.__stemmer = stemmer
        self.__lematizer = lematizer

    def __get_stemmer(self):
        return self.__stemmer

    stemmer = property(__get_stemmer)

    def __get_lematizer(self):
        return self.__lematizer

    lematizer = property(__get_lematizer)


    def stem(self, sent):
        return self.__stemmer.stem(sent)

    def lemmatize(self, sent):
        return self.__lematizer.lemmatize(sent)


    def tokenize(self, sent, transform = lambda s : s):
        pass

class RawSpacyWordMan(WordMan):

    def __init__(self, langRef : str, stemmLang : str, lematizer):
        super().__init__(SnowballStemmer(language=stemmLang), lematizer)
        self.__nlp = spacy.load(langRef)

    def rawTokenize(self, sent):
        doc = self.__nlp(sent)
        
        return [t.text for t in doc]

    def tokenize(self, sent, transform = lambda s : s):
        if transform is None: return self.rawTokenize(sent)

        doc = self.__nlp(sent)
        
        return [transform(t.text) for t in doc], [t.text for t in doc]

    
class DefaultWordMan(RawSpacyWordMan):

    def __init__(self, langRef : str, stemmLang : str, lematizer):
        super().__init__(langRef, stemmLang, lematizer)

    def __bringCloser(t : str, symbs : Iterable[str] = ["'"]):
        t1 = t
        for s in symbs:
            t2 = t1.replace(s + ' ', s)
            t3 = t1.replace(' ' + s, s)
            
            while t2 != t1 or t3 != t1:
                t1 = t2.replace(' ' + s, s)
                
                t2 = t1.replace(s + ' ', s)
                t3 = t1.replace(' ' + s, s)
                
        return t1

    def tokenize(self, sent, transform = lambda s : s.lower()):
        doc = DefaultWordMan.__bringCloser(sent)

        tokens = self.rawTokenize(doc)

        if transform is None: return tokens, tokens

        return [transform(w) for w in tokens], tokens


class EXtraTransformer:

    def __init__(self, accuracy, transform : Callable[[str], str]):
        self.__accuracy = accuracy
        self.__transform = transform


    def __get_accuracy(self):
        return self.__accuracy

    accuracy = property(__get_accuracy)

    def do(self, w: str):
        return self.__transform(w)


class ResponseSelector:

    def concate(entitySeq : Sequence[str]):
        return " ".join(entitySeq)

    def __init__(self, format = None):
        self._format = ResponseSelector.concate if format is None else format
        pass
    
    def select(self, preds):
        if len(preds) == 0 : return None

        it = iter(preds)
        fcls = next(it)
        fentityId = preds[fcls]['entityId']

        return self._format(fentityId), fcls, preds[fcls]

class NoNegSelector(ResponseSelector):

    def __init__(self, negWords, format=None):
        super().__init__(format)
        self.__negWords = negWords

    def select(self, preds) -> str:
        for cls in preds:
            fentityId = preds[cls]['entityId']
            
            ko = False
            for w in fentityId:
                if w in self.__negWords:
                    ko = True
                    break

            if ko: continue

            return self._format(fentityId), cls, preds[cls]
                
        return None

class EntityExtractor:

    def disorderedPartialPerfectTokensMatch(ref, test, et = None) -> bool:

        if et is None: et = EntityXtractTrainer.ET_IDENTITY
    
        tRef = [et.do(w) for w in ref]

        for r in test:
            if not (et.do(r) in tRef):return False
        
        return True

    def disorderedPerfectTokensMatch(ref, test, et = None) -> bool:
        nb = len(ref)
        if nb != len(test) : return False

        if et is None: et = EntityXtractTrainer.ET_IDENTITY
        
        tTest = [et.do(w) for w in test]
        
        for r in ref:
            if not (et.do(r) in tTest):return False
            
        return True

    def perfectTokensMatch(ref, test, et = None) -> bool:
        if len(ref) != len(test) : return False

        if et is None: et = EntityXtractTrainer.ET_IDENTITY
        
        for i, r in enumerate(ref):
            if et.do(r) != et.do(test[i]): return False
        
        return True

    def allowAllMatch(ref, test, et = None) -> bool:
        return True

    def extractName(tokens, np1:int, np2 :int) -> str:
        return tokens[np1: len(tokens) - np2 - 1]

    def __init__(self, estimator, wordMan, beforeWords, afterWords, negationWords, namePosDict):
        self.__estimator = estimator
        self.__wordMan = wordMan
        self.__beforeWords = beforeWords
        self.__afterWords = afterWords
        self.__namePosDict = namePosDict
        self.__negationWords = negationWords


    def __get_namePosDict(self):
        return self.__namePosDict

    namePosDict = property(__get_namePosDict)

    def __buildInputPatterns(self, tokens, maxWordLength = None):
        nbTokens = len(tokens)
    
        if nbTokens < 1: return []
        
        res = []
        
        if maxWordLength is None:
            maxWordLength = nbTokens
        elif maxWordLength > nbTokens: maxWordLength = nbTokens
            
        for nbWordLen in range(1, maxWordLength):
                
            for idx in range(0, nbTokens-nbWordLen+1):

                before = tokens[0: idx]
                after = tokens[idx + nbWordLen: ]

                entityId = tokens[idx: idx + nbWordLen]
                
                res.append((before, after, entityId))
        
        return res

    def __buildInputsFromPatterns(self, inputPatterns, et = None):
        if len(inputPatterns) == 0: return []
        if et is None: et = EntityXtractTrainer.ET_IDENTITY
        
        res = []
    
        for (before, after, entityId) in inputPatterns:
            tdocB = [et.do(w) for w in before]
            tdocA = [et.do(w) for w in after]

            nbMatchB = 0
            nbMatchA = 0
            
            features = []
            for w in self.__beforeWords:
                if w in tdocB:
                    features.append(1)
                    nbMatchB += 1
                else:
                    features.append(0)

            for w in self.__afterWords:
                if w in tdocA:
                    features.append(1)
                    nbMatchA += 1
                else:
                    features.append(0)

            nbMatch = nbMatchB + nbMatchA
            features.extend(EntityXtractTrainer.getPos(tdocB, tdocA))
            features.append(nbMatchB)
            features.append(nbMatch)
            features.extend(et.accuracy)

            if self.__negationWords is None:
                features.append(0)
                features.append(0)
            else:
                negB = 0
                negA = 0
                for w in self.__negationWords:
                    if et.do(w) in tdocB:
                        negB = 1
                        break

                for w in self.__negationWords:
                    if et.do(w) in tdocA:
                        negA = 1
                        break

                features.append(negB)
                features.append(negA)
            
            idInBefore = 0
            for w in entityId:
                if w in before:
                    idInBefore = 1
                    break

            idInAfter = 0
            for w in entityId:
                if w in after:
                    idInAfter = 1
                    break
            
            features.append(idInBefore)
            features.append(idInAfter)

            res.append(features)
                
        return res

    def __buildInputsFromTokens(self, tokens, maxNameLength = None, et = None):
        if et is None: et = EntityXtractTrainer.ET_IDENTITY

        inputPatterns = self.__buildInputPatterns(tokens, maxNameLength)
            
        return inputPatterns, self.__buildInputsFromPatterns(inputPatterns, et)


    def understandMessage(self, text, thresh, et = None, tokensMatch = None):
        if tokensMatch is None : tokensMatch = EntityExtractor.disorderedPartialPerfectTokensMatch

        tokens, rawTokens = self.__wordMan.tokenize(text)

        inputPatterns, gInputs = self.__buildInputsFromTokens(tokens, et = et)

        rawInputPatterns, _ = self.__buildInputsFromTokens(rawTokens, et = et)

        yPreds0 = []
    
        for eidx, gi in enumerate(gInputs):
            result = self.__estimator.predict(np.array([gi]))[0]

            for idx, res in enumerate(result):
                if res < thresh: continue

                npClass = self.__namePosDict[str(idx)]

                match = 0

                for i, bfr in enumerate(npClass['befores']):
                    if len(bfr) > 0:
                        
                        if not tokensMatch(bfr, inputPatterns[eidx][0], et):
                            match = -1
                            continue
                    elif len(inputPatterns[eidx][0]) > 0:
                        match = -1
                        continue

                    aft = npClass['afters'][i]

                    if len(aft) > 0:
                        if not tokensMatch(aft, inputPatterns[eidx][1], et):
                            match = -1
                            continue
                    elif len(inputPatterns[eidx][1]) > 0:
                        match = -1
                        continue
                        
                    match = 1
                    break

                if match < 0: continue

                #name = EntityExtractor.extractName(tokens, len(inputPatterns[eidx][0]), npClass['namePos'][1] + (len(inputPatterns[eidx][1]) - len(aft)))

                #name = inputPatterns[eidx][2]

                name = rawInputPatterns[eidx][2]

                if name is None or name == "": continue
                yPreds0.append((idx, res, name, eidx))

        yPreds0.sort(key=lambda x:x[1], reverse=True)

        yPreds1 = dict()

        for yp in yPreds0:
            if len(yp) == 0: continue
            cls = str(yp[0])

            if cls in yPreds1: continue
            yPreds1[cls] = { "p" : yp[1], "ip" : yp[3], "entityId" : yp[2] }
            
        return yPreds1, inputPatterns


    def selectResult(self, preds, selector = None):
        if selector is None : selector = NoNegSelector(self.__negationWords)
        return selector.select(preds)



class EntityXtractTrainer:

    ET_IDENTITY = EXtraTransformer([0, 0], lambda str : str)

    def __init__(self, wordMan = DefaultWordMan("fr_core_news_sm", "french", WordNetLemmatizer()), negationWords = None):
        self.__wordMan = wordMan
        self.__ET_STEM = EXtraTransformer([0, 1], lambda str : self.__wordMan.stem(str))
        self.__ET_LEM = EXtraTransformer([1, 0], lambda str : self.__wordMan.lemmatize(str))
        self.__negationWords = negationWords

    def __get_ET_LEM(self):
        return self.__ET_LEM

    ET_LEM = property(__get_ET_LEM)

    def __get_ET_STEM(self):
        return self.__ET_STEM

    ET_STEM = property(__get_ET_STEM)

    def __estimator(self, inputShape, outputShape):
        model = Sequential()
        model.add(Dense(128, input_shape=inputShape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(outputShape, activation="softmax"))

        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model


    def getPos(before, after):
        if len(before) == 0 and len(after) == 0:
            patternFeature = [1,1,1]
        elif len(before) == 0:
            patternFeature = [1,0,0]
        elif len(after) == 0:
            patternFeature = [0,0,1]
        else:
            patternFeature = [0,1,0]
            
        return patternFeature

    def __getTraingFeatures(self, idx, doc, docY : Sequence[object],  et = None):
        features = []

        if et is None: et = EntityXtractTrainer.ET_IDENTITY

        tdocB = [et.do(w) for w in doc[0]]
        tdocA = [et.do(w) for w in doc[1]]

        for w in self.__beforeWords:
            features.append(1 if et.do(w) in tdocB else 0)
            
        for w in self.__afterWords:
            features.append(1 if et.do(w) in tdocA else 0)
            
            
        nbMatch = len(tdocB) + len(tdocA)
        features.extend(doc[2])
        features.append(len(tdocB))
        features.append(nbMatch)
        features.extend(et.accuracy)

        if self.__negationWords is None:
            features.append(0)
            features.append(0)
        else:
            negB = 0
            negA = 0
            for w in self.__negationWords:
                if et.do(w) in tdocB:
                    negB = 1
                    break

            for w in self.__negationWords:
                if et.do(w) in tdocA:
                    negA = 1
                    break

            features.append(negB)
            features.append(negA)
        
        features.append(0)
        features.append(0)
    
        outputRow = [0] * len(self.__classes)
        outputRow[self.__classes.index(docY[idx])] = 1

        return features, outputRow
        
        #training.append([features, outputRow])


    def train(self, jsonData,  epochs=200):
        self.__beforeWords = []
        self.__afterWords = []

        docX = []
        docY = []
        classesDict = dict()
        namePosDict = dict()

        lastClassNum = 1

        classesDict["0"] = 0

        exps = jsonData['expresions']

        for sentenceConfig in exps:
            exp = sentenceConfig['exp']
        
            tokens, _ = self.__wordMan.tokenize(exp)

            namePos = sentenceConfig['namePos']
    
            isOK = sentenceConfig['ok']
            
            classKey = str(lastClassNum)

            before = tokens[0:namePos[0]]
            after = tokens[len(tokens) - namePos[1]-1:]
            
            classesDict[classKey] = lastClassNum
            namePosDict[str(lastClassNum)] = { 'namePos' : namePos, 'ok' : isOK, 'befores' : [before], 'afters' : [after]  } 
            lastClassNum += 1
            
            self.__beforeWords.extend(before)
            self.__afterWords.extend(after)

            patternFeature = EntityXtractTrainer.getPos(before, after)

            docX.append((before, after, patternFeature))
            docY.append(classesDict[classKey])

        stemB = [self.__wordMan.stem(w) for w in self.__beforeWords]
        stemA = [self.__wordMan.stem(w) for w in self.__afterWords]
        lemB = [self.__wordMan.lemmatize(w) for w in self.__beforeWords]
        lemA = [self.__wordMan.lemmatize(w) for w in self.__afterWords]

        self.__beforeWords = self.__beforeWords + stemB + lemB
        self.__afterWords = self.__afterWords + stemA + lemA

        self.__beforeWords = sorted(set(self.__beforeWords))
        self.__afterWords = sorted(set(self.__afterWords))

        self.__classes = sorted(set([classesDict[k] for k in classesDict]))

        training = []

        for idx, doc in enumerate(docX):
            features, outputRow = self.__getTraingFeatures(idx, doc, docY)
            training.append([features, outputRow])

            features, outputRow = self.__getTraingFeatures(idx, doc, docY, self.__ET_LEM)
            training.append([features, outputRow])

            features, outputRow = self.__getTraingFeatures(idx, doc, docY, self.__ET_STEM)
            training.append([features, outputRow])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        trainX = np.array(list(training[:, 0]))

        trainY = np.array(list(training[:, 1]))

        input_shape = (len(trainX[0]),)
        
        output_shape = (len(trainY[0]))

        print("Input size :", input_shape)
        print("Output size :", output_shape)

        model = self.__estimator(input_shape, output_shape)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        model.fit(x=trainX, y=trainY, epochs=epochs)

        return EntityExtractor(model, self.__wordMan, self.__beforeWords, self.__afterWords, self.__negationWords, namePosDict)
