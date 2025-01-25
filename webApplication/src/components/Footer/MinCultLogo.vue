<script setup>

</script>

<template>
    <div class="mincult">
        <img class="mincult__logo" alt="лого минкульта" src="/src/assets/images/mincult.png"/>
        <div class="mincult__text">
            Министерство культуры <br>Российской Федерации
        </div>
    </div>
</template>

<style scoped>
.mincult {
    display: flex;
    flex-direction: row;
    align-items: center;
}
.mincult__logo {
    display: block;
    height: 5.1875em;
    width: 4.6875em;
}
.mincult__text {
    font-size: var(--tertiary-font-size);
    margin-left: 1em;
    text-transform: uppercase;
    font-family: "Lora", serif;
    font-weight: 500;
    color: rgba(50, 45, 38, 0.8);
}
</style>